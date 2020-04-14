import random, json, os, logging
from collections import OrderedDict
from ctypes import Structure, c_float, c_byte, c_int, c_uint32, byref, cdll
from copy import deepcopy
from multiprocessing import Process, Queue, set_start_method
from multiprocessing.sharedctypes import RawArray

import numpy as np

from gym import spaces

_FLEX_BIN_PATH = ''
_DEFAULT_INIT_PARAMS = OrderedDict([
    ("renderBackend", 1),
    ("screenWidth", 1280),
    ("screenHeight", 720),
    ("msaaSamples", 4),
    ("device", -1)
])


class _NvFlexGymInitParams(Structure):
    _fields_ = [
        ("renderBackend", c_int),
        ("screenWidth", c_int),
        ("screenHeight", c_int),
        ("msaaSamples", c_int),
        ("device", c_int)
    ]


class FlexVecEnvSpec:
    def __init__(self, timestep_limit):
        self.timestep_limit = 1000
        self.id = 1


class FlexVecEnv:

    def __init__(self, flex_env_cfg):

        self._exp_cfg = flex_env_cfg['exp']
        self._num_envs = flex_env_cfg['scene']['NumAgents']

        # Load Flex Gym library
        self._flex_gym = make_flex_gym()
        init_params_dict = _DEFAULT_INIT_PARAMS.copy()

        if 'gym' in flex_env_cfg:
            init_params_dict.update(flex_env_cfg['gym'])

        init_params = _NvFlexGymInitParams(*init_params_dict.values())

        self._flex_gym.NvFlexGymInit(byref(init_params))
        self._flex_gym.NvFlexGymLoadScene(flex_env_cfg['scene_name'], json.dumps(flex_env_cfg['scene']))

        self._num_acts = self._flex_gym.NvFlexGymGetNumActions()
        self._num_obs = self._flex_gym.NvFlexGymGetNumObservations()

        if self.num_acts == -1 or self.num_obs == -1:
            raise ValueError(
                'FlexGym returned invalid num acts or obs! Got num_acts={}, num_obs={}'.format(self.num_acts,
                                                                                               self.num_obs))

        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1.)

        # Buffers
        self._total_acts = self.num_envs * self.num_acts
        self._act_buff = (c_float * self._total_acts)()

        self._total_obs = self.num_envs * self.num_obs
        self._obs_buff = (c_float * self._total_obs)()

        self._rew_buff = (c_float * self.num_envs)()
        self._death_buff = (c_byte * self.num_envs)()
        self._death_buff[:] = np.ones(self.num_envs).astype('byte')  # for first time reset

        self._default_infos = [{}] * self.num_envs
        self._alive_lengths = np.zeros(self.num_envs)

        self._first_reset = True

        self.spec = FlexVecEnvSpec(self._exp_cfg['max_ep_len'])

        logging.info("Done loading FlexVecEnv for scene {}".format(flex_env_cfg['scene_name']))

    def reset(self, agents_to_reset=None):
        """
        Reset all environments or just the done environments depending on agents_to_reset
        """
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
            self._death_buff[:] = np.zeros(self.num_envs).astype('byte')  # for first time reset

        return obs

    def step(self, actions):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        if self._first_reset:
            raise ValueError('Must call reset at least once before step!')

        self._act_buff[:] = actions.flatten()
        self._flex_gym.NvFlexGymSetActions(self._act_buff, 0, self._total_acts)
        for _ in range(self._sample_sim_steps()):
            quit = self._flex_gym.NvFlexGymUpdate()
            if quit != 0:
                self.close()
        self._alive_lengths += 1

        self._flex_gym.NvFlexGymGetObservations(self._obs_buff, 0, self._total_obs)
        self._flex_gym.NvFlexGymGetRewards(self._rew_buff, self._death_buff, 0, self.num_envs)

        obs = np.ctypeslib.as_array(self._obs_buff).reshape(self.num_envs, self.num_obs)
        rew = np.ctypeslib.as_array(self._rew_buff)
        die = np.ctypeslib.as_array(self._death_buff)

        # cap max ep len
        die[np.where(self._alive_lengths > self._exp_cfg['max_ep_len'])[0]] = 1

        return obs, rew, die, self._default_infos

    def _sample_sim_steps(self):

        frame_sampling_cfg = self._exp_cfg['frame_sampling']

        if frame_sampling_cfg['mode'] == 'constant':
            return frame_sampling_cfg['mean']
        elif frame_sampling_cfg['mode'] == 'geometric':
            return min(np.random.geometric(p=1. / frame_sampling_cfg['mean']), frame_sampling_cfg['max'])
        else:
            raise ValueError(
                'Unknown frame sampling mode {}! Need \'constant\' or \'geometric\''.format(frame_sampling_cfg['mode']))

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

    def set_seed(self, seed):
        self._flex_gym.NvFlexGymSetSeed(c_uint32(seed))

    def close(self):
        self._flex_gym.NvFlexGymShutdown()




def make_flex_vec_env(cfg):
    return FlexVecEnv(cfg)


def set_flex_bin_path(flex_bin_path):
    global _FLEX_BIN_PATH
    _FLEX_BIN_PATH = flex_bin_path


def make_flex_gym(debug=False):
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
