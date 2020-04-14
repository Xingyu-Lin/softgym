#!/usr/bin/env python
import argparse, logging, os, time
from autolab_core import YamlConfig

import horovod.tensorflow as hvd

import tensorflow as tf
import gym

from baselines import logger
from baselines.common import set_global_seeds, boolean_flag, tf_util as U
from baselines.ppo1.mlp_hvd_policy import MlpPolicy
from baselines.ppo1.clean_ppo1.PPO1_horovod import PPO1 as PPO1_distr_rollout
from baselines.ppo1.clean_ppo1.PPO1_distributed_training_horovod import PPO1 as PPO1_distr_train

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env


def train(logdir, cfg_env, cfg_train, no_distr_train):

    seed = cfg_train['seed'] + hvd.rank()
    set_global_seeds(seed)
    # has to be called BEFORE making FlexVecEnv, which changes the current directory
    logdir = os.path.realpath(logdir)
    os.makedirs(logdir, exist_ok=True)
    if no_distr_train:
        logger.configure(logdir)
    else:
        logger.configure(logdir, log_suffix='_{}'.format(hvd.rank()))
    cfg_env['gym']['seed'] = seed
    cfg_env['gpus'] = [hvd.local_rank()]
    # cfg_env['scene']['SimParamsStochastic'] = False

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = make_flex_vec_env(cfg_env)

    def policy_fn(name, ob_space, ac_space):
        try:
            policy_class = eval(cfg_train['policy_type'])
        except:
            logging.error('Policy type {} not found!'.format(cfg_train['policy_type']))
            exit()
        return policy_class(name=name, ob_space=ob_space, ac_space=ac_space, **cfg_train['policy'])

    gym.logger.setLevel(logging.WARN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    graph = tf.Graph()
    with graph.as_default():
        PPO1 = PPO1_distr_rollout if no_distr_train else PPO1_distr_train
        ppo = PPO1(logdir=logdir, **cfg_train['learn'])
        ppo.get_graph(policy_fn, env.num_envs, env.num_obs, env.num_acts)

    with tf.Session(config=config, graph=graph) as sess:
        ppo.learn(env, sess)
        U._PLACEHOLDER_CACHE = {}

    env.close()
    import gc
    gc.collect()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_rl', type=str, default='/sim2real/cfg/ppo1.yaml')
    parser.add_argument('--cfg_env', type=str, default='/sim2real/log/yumi_ropepeg')
    parser.add_argument('--no_distr_train', '-n', action='store_true')

    parser.add_argument('--rl_logdir', type=str, default='/sim2real/log/ppo1')

    parser.add_argument('--iter_flag', type=str, default='/sim2real/log/iter.flag')
    parser.add_argument('--run_rl_flag', type=str, default='/sim2real/log/run_rl.flag')
    parser.add_argument('--work_dir', type=str, default='')

    args = parser.parse_args()
    args.cfg_rl = args.work_dir + args.cfg_rl
    args.cfg_env = args.work_dir + args.cfg_env
    args.rl_logdir = args.work_dir + args.rl_logdir
    args.iter_flag = args.work_dir + args.iter_flag
    args.run_rl_flag = args.work_dir + args.run_rl_flag

    cwd_orig = os.getcwd()
    run_rl_flag = os.path.realpath(args.run_rl_flag)
    iter_flag = os.path.realpath(args.iter_flag)

    try:
        os.remove(run_rl_flag)
    except OSError:
        pass

    hvd.init()
    while True:
        print("Rank " + str(hvd.rank()) + ". Waiting for RL requests...")
        if os.path.isfile(run_rl_flag):
            with open(iter_flag, 'r') as f:
                cur_iter = f.read().replace('\n', '')
                f.close()
            print("Rank " + str(hvd.rank()) + ". RL request received. Iteration " + cur_iter)
            cfg_train = YamlConfig(args.cfg_rl)
            logdir = args.rl_logdir + "_" + cur_iter
            cfg_env_path = args.cfg_env + "_" + cur_iter + ".yaml"
            cfg_env = YamlConfig(cfg_env_path)
            train(logdir, cfg_env, cfg_train, args.no_distr_train)

            os.chdir(cwd_orig)
            try:
                os.remove(run_rl_flag)
            except OSError:
                pass

            print("Rank " + str(hvd.rank()) + ". Finished RL. Iteration " + cur_iter)
        time.sleep(2.0)
