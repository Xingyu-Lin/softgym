#!/usr/bin/env python
import argparse, logging, os
from time import time
import numpy as np
from autolab_core import YamlConfig

from baselines import bench, logger
from baselines.common import set_global_seeds, tf_util as U

import baselines.ddpg.training_par_vecenv as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.ddpg.memory_par import Memory
from baselines.ddpg.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.ddpg.prioritized_buffer import PrioritizedBuffer
from baselines.ddpg.unprioritized_buffer import UnprioritizedBuffer

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env

def train(logdir, cfg_env, cfg_train):
    set_global_seeds(cfg_train['seed'])
    # has to be called BEFORE making FlexVecEnv, which changes the current directory
    logdir = os.path.realpath(logdir)
    os.makedirs(logdir, exist_ok=True)
    logger.configure(logdir)
    sess = U.single_threaded_session()
    sess.__enter__()

    cfg_env['gym']['seed'] = cfg_train['seed']
    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = make_flex_vec_env(cfg_env)

    # Parse noise_type
    action_noise = None
    param_noise = None
    if not cfg_train['learn']['noisy_net']:
        for current_noise_type in cfg_train['noise_type'].split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros((env.num_envs, env.num_acts)),
                                                sigma=float(stddev) * np.ones((env.num_envs, env.num_acts)))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros((env.num_envs, env.num_acts)),
                                                            sigma=float(stddev) * np.ones((env.num_envs, env.num_acts)))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    if cfg_train['learn']['prioritized_replay']:
        memory = PrioritizedReplayBuffer(int(1e6), alpha=cfg_train['learn']['prioritized_replay_alpha'])
        #memory = PrioritizedBuffer(int(1e6), prioritized_replay_alpha)
    else:
        memory = UnprioritizedBuffer(int(1e6))
        #memory = Memory(int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
        #memory = ReplayBuffer(int(1e6))


    critic = Critic(**cfg_train['policy'])
    actor = Actor(env.num_acts, **cfg_train['policy'])

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(cfg_train['seed'], logger.get_dir()))
    set_global_seeds(cfg_train['seed'])

    start_time = time()
    training.train(env=env, logdir=logdir, num_parallel=env.num_envs, param_noise=param_noise,
                   action_noise=action_noise, actor=actor, critic=critic, memory=memory, 
                   # rendering will be handled by yaml cfg
                   render=False, render_eval=False, **cfg_train['learn'])
                   

    logger.Logger.CURRENT.close()
    logger.info('total runtime: {}s'.format(time() - start_time))

    env.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs/ddpg')
    parser.add_argument('--cfg_env', type=str, default='cfg/ant.yaml')
    parser.add_argument('--cfg_train', type=str, default='cfg/train/ddpg.yaml')
    args = parser.parse_args()

    cfg_env = YamlConfig(args.cfg_env)
    cfg_train = YamlConfig(args.cfg_train)

    train(args.logdir, cfg_env, cfg_train)