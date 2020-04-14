#!/usr/bin/env python
import argparse, logging, os
import numpy as np
import tensorflow as tf
from autolab_core import YamlConfig
import gym

from baselines import bench, logger
from baselines.common import set_global_seeds, boolean_flag, tf_util as U
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy

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

    policy = MlpPolicy
    gym.logger.setLevel(logging.WARN)
    ppo2.learn(policy=policy, policy_params=cfg_train['policy'], env=env, logdir=logdir, **cfg_train['learn'])

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs/ppo2')
    parser.add_argument('--cfg_env', type=str, default='cfg/ant.yaml')
    parser.add_argument('--cfg_train', type=str, default='cfg/train/ppo2.yaml')
    args = parser.parse_args()

    cfg_env = YamlConfig(args.cfg_env)
    cfg_train = YamlConfig(args.cfg_train)

    train(args.logdir, cfg_env, cfg_train)
