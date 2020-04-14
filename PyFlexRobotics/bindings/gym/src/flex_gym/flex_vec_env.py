import random, json, os, logging
from collections import OrderedDict
from ctypes import Structure, c_float, c_byte, c_int, c_uint32, c_bool, byref, cdll
from copy import deepcopy
from multiprocessing import Process, Queue, set_start_method
from multiprocessing.sharedctypes import RawArray
from time import time
import sys, pdb

import numpy as np
import GPUtil

from gym import spaces

_FLEX_BIN_PATH = ''
_DEFAULT_INIT_PARAMS = OrderedDict([
            ('renderBackend', 1),
            ('screenWidth', 1280),
            ('screenHeight', 720),
            ('msaaSamples', 4),
            ('device', -1),
            ('rank', 0),
            ('seed', 1),
            ('vsync', False)
        ])

class _NvFlexGymInitParams(Structure):
    _fields_ = [
        ('renderBackend', c_int),
        ('screenWidth', c_int),
        ('screenHeight', c_int),
        ('msaaSamples', c_int),
        ('device', c_int),
        ('rank', c_int),
        ('seed', c_uint32),
        ('vsync', c_bool)
    ]

class FlexVecEnvSpec:
    def __init__(self, timestep_limit):
        self.timestep_limit = timestep_limit
        self.id = 1

class FlexVecEnv:

    def __init__(self, flex_env_cfg):
        self._cfg = flex_env_cfg
        self._exp_cfg = flex_env_cfg['exp']
        self._num_envs = flex_env_cfg['scene']['NumAgents']
        if 'InitialGrasp' in flex_env_cfg['scene']:
            if flex_env_cfg['scene']['DoGripperControl']:
                self._initial_grasp = flex_env_cfg['scene']['InitialGrasp']
                self._initial_grasp_probability = flex_env_cfg['scene']['InitialGraspProbability']
                self._relative_goals = flex_env_cfg['scene']['RelativeTarget']
                self._wrist_control = flex_env_cfg['scene']['DoWristRollControl']
            else:
                self._initial_grasp = None
        else:
            self._initial_grasp = None

        # Load Flex Gym library
        self._flex_gym = _load_flex_gym()
        init_params_dict = _DEFAULT_INIT_PARAMS.copy()
        if 'gym' in flex_env_cfg:
            init_params_dict.update(flex_env_cfg['gym'])
        init_params = _NvFlexGymInitParams(*init_params_dict.values())
        self._flex_gym.NvFlexGymInit(byref(init_params))
        self._flex_gym.NvFlexGymLoadScene(flex_env_cfg['scene_name'], json.dumps(flex_env_cfg['scene']))

        self._num_acts = self._flex_gym.NvFlexGymGetNumActions()
        self._num_obs = self._flex_gym.NvFlexGymGetNumObservations()
        self._num_extras = self._flex_gym.NvFlexGymGetNumExtras()
        self._num_pytoc = self._flex_gym.NvFlexGymGetNumPyToC()

        if self.num_acts == -1 or self.num_obs == -1:
            raise ValueError('FlexGym returned invalid num acts or obs! Got num_acts={}, num_obs={}'.format(self.num_acts, self.num_obs))

        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1.)

        # Buffers
        self._total_acts = self.num_envs * self.num_acts
        self._act_buff = (c_float * self._total_acts)()

        self._total_obs = self.num_envs * self.num_obs
        self._obs_buff = (c_float * self._total_obs)()

        self._rew_buff = (c_float * self.num_envs)()
        self._death_buff = (c_byte * self.num_envs)()
        self._death_buff[:] = np.ones(self.num_envs).astype('byte') # for first time reset

        self._total_extras = self.num_envs * self.num_extras
        self._extras_buff = (c_float * self._total_extras)()

        self._total_pytoc = self.num_envs * self._num_pytoc
        self._pytoc_buff = (c_float * self._total_pytoc)()

        self._alive_lengths = np.zeros(self.num_envs)

        self._first_reset = True
        self._closed = False

        self.spec = FlexVecEnvSpec(self._exp_cfg['max_ep_len'])

        logging.info('Done loading FlexVecEnv for scene {}'.format(flex_env_cfg['scene_name']))

    def unscale(self, lo, hi, y):
        return 2. * (y - lo) / (hi - lo) - 1.

    def unscale_action(self, target_pose):
        return np.swapaxes(np.array([
            self.unscale(-0.5, 0.5, target_pose[:, 0]),
            self.unscale(0.4, 0.8, target_pose[:, 1]),
            self.unscale(0.0, 1.0, target_pose[:, 2]),
            self.unscale(0.022, 0.05, target_pose[:, 3])
        ]),0,1)

    def hardcoded_reset(self, obs):
        """
        This function is written to allow resetting an agent such that it is grasping the object that needs
        to be manipulated. Tested with RL Fetch environments to use with delta planar control. 
        Look at fetch_cube_hardcoded.py for a full DoF example.
        """

        open_width = 0.05
        grasp_width = 0.023;  # Change this depending on the object you are trying to manipulate

        prep_pose = np.zeros((self.num_envs, self._num_acts))
        reach_pose = np.zeros((self.num_envs, self._num_acts))
        grasp_pose = np.zeros((self.num_envs, self._num_acts))
        gripper_loc = np.zeros((self.num_envs, 3))
        
        if(self._relative_goals):
            object_loc = obs[:, self.num_acts:self.num_acts+3] + obs[:, :3]  # The addition term is the end-effector position
        else:
            object_loc = obs[:, self.num_acts:self.num_acts+3]

        for i in range(self.num_envs):
            prep_pose[i] = np.r_[object_loc[i] + np.array([0.0, 0.2, 0.0]),[open_width]]
            reach_pose[i] = np.r_[object_loc[i] + np.array([0.0, 0.11, 0.0]),[open_width]]
            grasp_pose[i] = np.r_[object_loc[i] + np.array([0.0, 0.11, 0.0]),[grasp_width]]

        state = ['prep' for _ in range(self.num_envs)]
        target_pose = prep_pose

        s = time()
        while True:
            sys.stdout.flush()
            act = self.unscale_action(target_pose)

            obs, rew, done, _ = self.step(np.array([act]))

            gripper_loc[:] = obs[:, :3]
            # logging.info('reward: {:.3f}'.format(rew[0]))

            if time() - s < 1:
                continue

            for j in range(self.num_envs):

                if state[j] == 'prep':
                    if gripper_loc[j, 0] < prep_pose[j, 0]:
                        # switching to reach
                        state[j] = 'reach'
                        target_pose[j] = reach_pose[j]

                elif state[j] == 'reach':
                    if gripper_loc[j, 1] < 0.46:
                        # switching to grasp
                        state[j] = 'grasp'
                        target_pose[j] = grasp_pose[j]

                elif state[j] == 'grasp':
                    if obs[j, self.num_acts-1] < grasp_width + 0.0005:
                        # switching to RL agent control
                        return

    def reset(self, agents_to_reset=None, get_info=False):
        '''
        Reset all environments or just the done environments depending on agents_to_reset
        '''
        if self._closed:
            raise ValueError('Env has been closed!')
        if agents_to_reset is None:
            agents_to_reset = np.where(np.ctypeslib.as_array(self._death_buff) == 1)[0]

        if len(agents_to_reset) == self.num_envs:
            self._flex_gym.NvFlexGymResetAllAgents()
        else:
            for agent in agents_to_reset:
                self._flex_gym.NvFlexGymResetAgent(int(agent))
        self._alive_lengths[agents_to_reset] = 0

        self._flex_gym.NvFlexGymGetObservations(self._obs_buff, 0, self._total_obs)
        obs = np.ctypeslib.as_array(self._obs_buff).reshape(self.num_envs, self.num_obs)

        if self._first_reset:
            self._first_reset = False
            self._death_buff[:] = np.zeros(self.num_envs).astype('byte') # for first time reset

        if get_info:
            self._flex_gym.NvFlexPopulateExtras()
            self._flex_gym.NvFlexGymGetExtras(self._extras_buff, 0, self._total_extras)
            infos = np.ctypeslib.as_array(self._extras_buff).reshape(self.num_envs, self.num_extras)

            if self._initial_grasp:
                if(random.random() < self._initial_grasp_probability):
                    self.hardcoded_reset(obs)
                    self._flex_gym.NvFlexGymGetObservations(self._obs_buff, 0, self._total_obs)
                    obs = np.ctypeslib.as_array(self._obs_buff).reshape(self.num_envs, self.num_obs)
                    self._flex_gym.NvFlexPopulateExtras()
                    self._flex_gym.NvFlexGymGetExtras(self._extras_buff, 0, self._total_extras)
                    infos = np.ctypeslib.as_array(self._extras_buff).reshape(self.num_envs, self.num_extras)

            return obs, infos

        if self._initial_grasp:
            if(random.random() < self._initial_grasp_probability):
                self.hardcoded_reset(obs)
                self._flex_gym.NvFlexGymGetObservations(self._obs_buff, 0, self._total_obs)
                obs = np.ctypeslib.as_array(self._obs_buff).reshape(self.num_envs, self.num_obs)

        return obs

    def step(self, actions, times=None):
        '''
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        '''
        if self._closed:
            raise ValueError('Env has been closed!')
        if self._first_reset:
            raise ValueError('Must call reset at least once before step!')

        self._act_buff[:] = actions.flatten()
        self._flex_gym.NvFlexGymSetActions(self._act_buff, 0, self._total_acts)
        for _ in range(self._sample_sim_steps()):
            if times is not None:
                start_time = time()
            quit = self._flex_gym.NvFlexGymUpdate()
            if times is not None:
                times.append(time() - start_time)
            if quit != 0:
                self.close()
        self._alive_lengths += 1

        self._flex_gym.NvFlexGymGetObservations(self._obs_buff, 0, self._total_obs)
        self._flex_gym.NvFlexGymGetRewards(self._rew_buff, self._death_buff, 0, self.num_envs)
        if (self.num_extras > 0):
            self._flex_gym.NvFlexGymGetExtras(self._extras_buff, 0, self._total_extras)

        obs = np.ctypeslib.as_array(self._obs_buff).reshape(self.num_envs, self.num_obs)
        rew = np.ctypeslib.as_array(self._rew_buff)
        die = np.ctypeslib.as_array(self._death_buff)
        if (self.num_extras > 0):
            infos = np.ctypeslib.as_array(self._extras_buff).reshape(self.num_envs, self.num_extras)

        # cap max ep len
        die[np.where(self._alive_lengths > self._max_episode_steps)[0]] = 1

        if (self.num_extras > 0):
            return obs, rew, die, infos
        else:
            return obs, rew, die, None

    def _sample_sim_steps(self):
        frame_sampling_cfg = self._exp_cfg['frame_sampling']
        if frame_sampling_cfg['mode'] == 'constant':
            return frame_sampling_cfg['mean']
        elif frame_sampling_cfg['mode'] == 'geometric':
            return min(np.random.geometric(p = 1./frame_sampling_cfg['mean']), frame_sampling_cfg['max'])
        else:
            raise ValueError('Unknown frame sampling mode {}! Need \'constant\' or \'geometric\''.format(frame_sampling_cfg['mode']))

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_obs(self):
        return self._num_obs

    @property
    def num_acts(self):
        return self._num_acts

    @property
    def num_extras(self):
        return self._num_extras

    @property
    def num_obs_normalize(self):
        if self._num_obs > 1000:
            return self._num_obs - 64*64*4
        else:
            return self._num_obs

    @property
    def num_goal(self):
        return (self.num_extras-1)//2

    @property
    def num_goal_normalize(self):
        if self.num_goal > 1000:
            return self.num_goal - 64*64*4
        else:
            return self.num_goal

    def compute_reward(self, achieved_goal, desired_goal, info):
        if 'LR' in self._cfg['scene_name']:
            desired_direction = (desired_goal[:,0] > 0.0).astype(np.float32)*2 - 1.0
            achieved_direction = (achieved_goal[:,0] > 0.0).astype(np.float32)*2 - 1.0
            rewards = achieved_direction * desired_direction
        else:
            distance_threshold = 0.1
            dists = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            rewards = np.ones(len(achieved_goal)) * -1
            rewards[np.where(dists <= distance_threshold)[0]] = 0
        return rewards

    @property
    def _max_episode_steps(self):
        return self._exp_cfg['max_ep_len']

    def seed(self, s):
        random.seed(s)
        self.set_seed(s)

    def set_seed(self, seed):
        self._flex_gym.NvFlexGymSetSeed(c_uint32(seed))

    def close(self):
        self._flex_gym.NvFlexGymShutdown()
        self._closed = True

class _FlexVecEnvMultiEnvWorker(Process):

    def __init__(self, cfg, i, flex_bin_path, arr_acts, arr_obs, arr_rews, arr_dones, arr_resets, cmd_q, done_q):
        super().__init__()
        self._cfg = cfg
        self._i = i
        self._flex_bin_path = flex_bin_path
        
        self._arr_acts = arr_acts
        self._arr_obs = arr_obs
        self._arr_rews = arr_rews
        self._arr_dones = arr_dones
        self._arr_resets = arr_resets
        
        self._cmd_q = cmd_q
        self._done_q = done_q

    def run(self):
        set_flex_bin_path(self._flex_bin_path)
        self._arr_acts = np.frombuffer(self._arr_acts, dtype=np.float32)
        self._arr_obs = np.frombuffer(self._arr_obs, dtype=np.float32)
        self._arr_rews = np.frombuffer(self._arr_rews, dtype=np.float32)
        self._arr_dones = np.frombuffer(self._arr_dones, dtype=np.byte)
        self._arr_resets = np.frombuffer(self._arr_resets, dtype=np.byte)
        
        self._env = FlexVecEnv(self._cfg)

        i_range = np.array([self._i, self._i + 1])
        self._range_acts =  i_range * self._env.num_acts * self._env.num_envs
        self._range_obs = i_range * self._env.num_obs * self._env.num_envs
        self._range_rews = i_range * self._env._num_envs
        self._range_dones = i_range * self._env._num_envs
        self._range_resets = i_range * self._env._num_envs

        self._done_q.put(True)

        while True:
            cmd = self._cmd_q.get()
            if cmd == 'reset_None':
                obs = self._env.reset()
                self._set_obs(obs)
            elif cmd == 'reset':
                agents_to_reset = self._get_agent_resets()
                obs = self._env.reset(agents_to_reset=agents_to_reset)
                self._set_obs(obs)
            elif cmd == 'step':
                acts = self._get_acts()
                obs, rews, dones, _ = self._env.step(acts)
                self._set_obs(obs)
                self._set_dones(dones)
                self._set_rews(rews)
            elif cmd == 'set_seed':
                seed = self._cmd_q.get()
                self._env.set_seed(seed)
            elif cmd == 'close':
                self._env.close()

            self._done_q.put(True)
            if cmd == 'close':
                break

    def _get_acts(self):
        return self._arr_acts[self._range_acts[0] : self._range_acts[1]]

    def _get_agent_resets(self):
        return self._arr_resets[self._range_obs[0] : self._range_obs[1]]

    def _set_dones(self, dones):
        self._arr_dones[self._range_dones[0] : self._range_dones[1]] = dones

    def _set_obs(self, obs):
        self._arr_obs[self._range_obs[0] : self._range_obs[1]] = obs.flatten()

    def _set_rews(self, rews):
        self._arr_rews[self._range_rews[0] : self._range_rews[1]] = rews

class FlexVecEnvMultiEnv:

    def __init__(self, flex_env_cfgs):
        self._workers = []
        num_workers = len(flex_env_cfgs) - 1
        self._cmd_qs = [Queue() for _ in range(num_workers)]
        self._done_qs = [Queue() for _ in range(num_workers)]
        
        self._num_agents = []

        for i, cfg in enumerate(flex_env_cfgs):
            if i == 0:
                self._env = FlexVecEnv(cfg) 
                self._num_agents.append(cfg['scene']['NumAgents'])
                self._scene_name = cfg['scene_name']
                
                self._arr_acts = RawArray('f', self._env.num_acts * self._env.num_envs * num_workers)
                self._arr_obs = RawArray('f', self._env.num_obs * self._env.num_envs * num_workers)
                self._arr_rews = RawArray('f', self._env.num_envs * num_workers)
                self._arr_dones = RawArray('b', self._env.num_envs * num_workers)
                self._arr_resets = RawArray('b', self._env.num_envs * num_workers)
            else:
                if cfg['scene_name'] != self._scene_name:
                    raise ValueError('MultiEnv does not support multiple scene types!')
                self._num_agents.append(cfg['scene']['NumAgents'])
                self._workers.append(_FlexVecEnvMultiEnvWorker(cfg, i - 1, _FLEX_BIN_PATH, 
                    self._arr_acts, self._arr_obs, self._arr_rews, self._arr_dones, self._arr_resets,
                    self._cmd_qs[i - 1], self._done_qs[i - 1]
                ))
            
        self._arr_acts = np.frombuffer(self._arr_acts, dtype=np.float32)
        self._arr_obs = np.frombuffer(self._arr_obs, dtype=np.float32)
        self._arr_rews = np.frombuffer(self._arr_rews, dtype=np.float32)
        self._arr_dones = np.frombuffer(self._arr_dones, dtype=np.byte)
        self._arr_resets = np.frombuffer(self._arr_resets, dtype=np.byte)

        logging.info('Starting {} FlexVecEnv workers'.format(len(flex_env_cfgs)))
        for worker in self._workers:
            worker.start()
        for q in self._done_qs:
            q.get()
        logging.info('All FlexVecEnv workers started!')

    def step(self, actions):
        self._arr_acts[:] = actions[self._env.num_envs:].flatten()

        self._cmd_fork('step')
        obs, rews, dones, _ = self._env.step(actions[:self._env.num_envs])
        self._cmd_join()

        obs = self._get_obs(obs)
        rews = np.concatenate([rews, self._arr_rews])
        dones = np.concatenate([dones, self._arr_dones])

        return obs, rews, dones, []

    def reset(self, agents_to_reset=None):
        if agents_to_reset is None:
            self._cmd_fork('reset_None')
            obs = self._env.reset()
            self._cmd_join()
        else:
            agents_to_reset = np.sort(agents_to_reset)
            first_agents_to_reset = []
            for ai in agents_to_reset:
                if ai < self.num_envs:
                    first_agents_to_reset.append(ai)
                else:
                    break

            rest_agents_to_reset = np.zeros((self._workers - 1) * self._env.num_envs)
            for i in range(len(first_agents_to_reset), len(agents_to_reset)):
                ai = agents_to_reset[i]
                rest_agents_to_reset[i - self._env.num_envs] = 1
            self._arr_resets[:] = rest_agents_to_reset
            
            self._cmd_fork('reset')
            obs = self._env.reset(first_agents_to_reset)
            self._cmd_join()

        return self._get_obs(obs)

    def replace_env(self, i, cfg):
        if self._num_agents[i] != cfg['scene']['NumAgents']:
            raise ValueError('Cannot change the number of agents from the original to new env!')
        if self._scene_name != cfg['scene_name']:
            raise ValueError('Cannot change scene from original to new env!')

        if i == 0:
            self._env.close()
            self._env = FlexVecEnv(cfg)
        else:
            j = i - 1
            self._cmd_qs[j].put('close')
            self._done_qs[j].get()
            self._workers[j] = _FlexVecEnvMultiEnvWorker(cfg, j, _FLEX_BIN_PATH, 
                    self._arr_acts, self._arr_obs, self._arr_rews, self._arr_dones, self._arr_resets,
                    self._cmd_qs[j], self._done_qs[j]
                )
            self._workers[j].start()

    def _cmd_fork(self, cmd):
        for q in self._cmd_qs:
            q.put(cmd)

    def _cmd_join(self):
        for q in self._done_qs:
            q.get()

    def _get_obs(self, obs):
        return np.concatenate([obs, self._arr_obs.reshape(-1, self.num_obs)])

    @property
    def action_space(self):
        return self._env._action_space

    @property
    def observation_space(self):
        return self._env._observation_space

    @property
    def num_envs(self):
        return (len(self._workers) + 1) * self._env.num_envs

    @property
    def num_obs(self):
        return self._env.num_obs

    @property
    def num_acts(self):
        return self._env.num_acts

    def set_seed(self, seed):
        self._cmd_fork('set_seed')
        self._cmd_fork(seed)
        self._env.set_seed(seed)
        self._cmd_join()

    def close(self):
        self._cmd_fork('close')
        self._env.close()
        self._cmd_join()
    
class FlexVecEnvMultiGPU(FlexVecEnvMultiEnv):

    def __init__(self, flex_env_cfg, gpu_device_ids):
        self._ensure_valid_gpus(gpu_device_ids)
        cfgs = []
        for i, id in enumerate(gpu_device_ids):
            cfg = deepcopy(flex_env_cfg)
            cfg['gym']['device'] = id
            if 'seed' not in cfg['gym']:
                cfg['gym']['seed'] = _DEFAULT_INIT_PARAMS['seed']
            cfg['gym']['seed'] += i
            cfgs.append(cfg)

        super().__init__(cfgs)

    def _ensure_valid_gpus(self, gpu_device_ids):
        valid_gpu_ids = set([gpu.id for gpu in GPUtil.getGPUs()])
        for id in gpu_device_ids:
            if id not in valid_gpu_ids:
                raise ValueError('GPU ID {} not found!'.format(id))

def make_flex_vec_env(cfg):
    if 'gpus' in cfg:
        if isinstance(cfg['gpus'], int):
            available_gpus = GPUtil.getGPUs()
            if len(available_gpus) < cfg['gpus']:
                raise ValueError('System has {} GPUs, which is less than the {} specified in config!'.format(len(available_gpus), cfg['gpus']))
            gpus = [gpu.id for gpu in available_gpus][:cfg['gpus']]
        elif isinstance(cfg['gpus'], list):
            gpus = cfg['gpus']
        else:
            raise ValueError('Invalid gpus param in env cfg! Expecting list of ints or int!')
        
        if len(gpus) > 1:
            logging.info('Making Multi-GPU FlexVecEnv w/ GPUs {}'.format(gpus))
            set_start_method('spawn', force=True)
            return FlexVecEnvMultiGPU(cfg, gpus)
        elif len(gpus) == 1:
            cfg['gym']['device'] = gpus[0]
        else:
            raise ValueError('Cannot create Flex w/ 0 GPUs! Did you mean gpus: [0]?')
    return FlexVecEnv(cfg)

def make_flex_vec_env_muli_env(cfgs):
    for cfg in cfgs:
        if 'gpus' in cfg:
            if isinstance(cfg['gpus'], int):
                if cfg['gpus'] > 1:
                    raise ValueError('Multi Env does not support multiple GPUs per env!')
                cfg['gym']['device'] = 0
            elif isinstance(cfg['gpus'], list):
                if len(cfg['gpus']) > 1:
                    raise ValueError('Multi Env does not support multiple GPUs per env!')
                cfg['gym']['device'] = cfg['gpus'][0]
            else:
                raise ValueError('Invalid gpus param in env cfg! Expecting list of ints or int!')
    set_start_method('spawn', force=True)
    return FlexVecEnvMultiEnv(cfgs)

def set_flex_bin_path(flex_bin_path):
    global _FLEX_BIN_PATH
    _FLEX_BIN_PATH = flex_bin_path

def _load_flex_gym(debug=False):
    if _FLEX_BIN_PATH == '':
        raise ValueError('Must call set_flex_bin_path first!')

    on_windows = os.name == 'nt'

    flex_gym_path = os.path.join(_FLEX_BIN_PATH, 'win64' if on_windows else 'linux64')
    os.chdir(flex_gym_path)
    return cdll.LoadLibrary(os.path.join(flex_gym_path,
            'NvFlexGym{}CUDA_x64{}'.format(
                'Debug' if debug else 'Release',
                '' if on_windows else '.so'
            ))
        )
