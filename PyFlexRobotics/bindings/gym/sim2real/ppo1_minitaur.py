#!/usr/bin/env python
import argparse
import logging
import os

import gym

from autolab_core import YamlConfig

from baselines import bench, logger
from baselines.common import set_global_seeds, boolean_flag, tf_util as U
from baselines.ppo1 import pposgd_vecenv_minitaur
from baselines.ppo1.mlp_policy import MlpPolicy, MlpBetaPolicy, MlpRaycastPolicy, MlpRaycastCNNPolicy
from baselines.ppo1.noisy_policies import MlpNoisyPolicy
from baselines.ppo1.ppo1_curriculum_cfg import CurriculumConfigSwitcher

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env

from simtaur import SimtaurScene


def train(simtaur_scene, logdir, cfg_env, cfg_train, cfg_curr=None):
    set_global_seeds(cfg_train['seed'])
    # has to be called BEFORE making FlexVecEnv, which changes the current directory
    logdir = os.path.realpath(logdir)
    os.makedirs(logdir, exist_ok=True)
    logger.configure(logdir)
    sess = U.single_threaded_session()
    sess.__enter__()

    cfg_env['gym']['seed'] = cfg_train['seed']

    set_flex_bin_path(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../../../bin'))
    env = sim_scene.env

    # for curriculum config switching
    callback = None
    if cfg_curr:
        curr_switcher = CurriculumConfigSwitcher(
            cfg_env, cfg_curr, make_flex_vec_env)
        callback = curr_switcher.get_callback()

    def policy_fn(name, ob_space, ac_space):
        try:
            policy_class = eval(cfg_train['policy_type'])
        except:
            logging.error('Policy type {} not found!'.format(
                cfg_train['policy_type']))
            exit()
        return policy_class(name=name, ob_space=ob_space, ac_space=ac_space, **cfg_train['policy'])

    gym.logger.setLevel(logging.WARN)
    pposgd_vecenv_minitaur.learn(env, policy_fn, simtaur_scene=simtaur_scene,
                                 logdir=logdir, callback=callback, **cfg_train['learn'])
    env.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str,
                        default='/flex_gym_io/logs/ppo1_minitaur_torque')
    parser.add_argument('--cfg_env', type=str, default='cfg/minitaur.yaml')
    parser.add_argument('--cfg_train', type=str,
                        default='cfg/train/ppo1_minitaur.yaml')
    parser.add_argument('--cfg_curriculum', type=str, default='')
    args = parser.parse_args()

    cfg_env = YamlConfig(args.cfg_env)
    cfg_train = YamlConfig(args.cfg_train)

    cfg_curr = None
    if args.cfg_curriculum:
        cfg_curr = YamlConfig(args.cfg_curriculum)

    sim_scene = SimtaurScene()
    # sim_scene.loopFlexGym()

    # Waiting for env to start
    while sim_scene.env == None:
        pass

    train(sim_scene, args.logdir, cfg_env, cfg_train, cfg_curr)
