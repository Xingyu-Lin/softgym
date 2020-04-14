#!/usr/bin/env python
import argparse, logging, os
import gym
from gym import utils
from autolab_core import YamlConfig
import tensorflow as tf

from baselines import bench, logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont_parallel import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

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

    gym.logger.setLevel(logging.WARN)

    with tf.variable_scope("vf"):
        vf = NeuralNetValueFunction(env.num_obs, env.num_acts, optim_epochs=cfg_train['learn']['optim_epochs'], **cfg_train['policy'])
    with tf.variable_scope("pi"):
        policy = GaussianMlpPolicy(env.num_obs, env.num_acts, **cfg_train['policy'])
    learn(env,  num_parallel=env.num_envs, policy=policy, vf=vf, logdir=logdir, **cfg_train['learn'])

    env.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs/acktr')
    parser.add_argument('--cfg_env', type=str, default='cfg/ant.yaml')
    parser.add_argument('--cfg_train', type=str, default='cfg/train/acktr.yaml')
    args = parser.parse_args()

    cfg_env = YamlConfig(args.cfg_env)
    cfg_train = YamlConfig(args.cfg_train)

    train(args.logdir, cfg_env, cfg_train)
