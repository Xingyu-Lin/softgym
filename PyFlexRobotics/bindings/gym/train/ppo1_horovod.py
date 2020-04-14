import argparse, logging, os

import horovod.tensorflow as hvd
hvd.init()
import tensorflow as tf
import gym
from autolab_core import YamlConfig

from baselines import bench, logger
from baselines.common import set_global_seeds, boolean_flag, tf_util as U
from baselines.ppo1.mlp_hvd_policy import MlpPolicy, MlpBetaPolicy, MlpRaycastPolicy, MlpRaycastCNNPolicy
from baselines.ppo1.clean_ppo1.PPO1_horovod import PPO1 as PPO1_distr_rollout
from baselines.ppo1.clean_ppo1.PPO1_distributed_training_horovod import PPO1 as PPO1_distr_train
from baselines.ppo1.ppo1_curriculum_cfg import CurriculumConfigSwitcher
from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env

def train(logdir, cfg_env, cfg_train, no_distr_train, cfg_curr=None):
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
    cfg_env['gym']['rank'] = hvd.rank()
    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = make_flex_vec_env(cfg_env)

    callback = None
    if cfg_curr:
        for vals in cfg_curr.config.values():
            if 'scene' in vals['cfg']:
                if 'NumAgents' in vals['cfg']['scene']:
                    raise ValueError('PPO1 Horovod does not support variable agents during curriculum training!')
        curr_switcher = CurriculumConfigSwitcher(cfg_env, cfg_curr, make_flex_vec_env)
        callback = curr_switcher.get_callback()

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
        ppo = PPO1(logdir=logdir, callback=callback, **cfg_train['learn'])
        ppo.get_graph(policy_fn, env.num_envs, env.num_obs, env.num_acts)

    with tf.Session(config=config, graph=graph) as sess:
        ppo.learn(env, sess)

    env.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_distr_train', '-n', action='store_true')
    parser.add_argument('--multi_env_cfg_path', '-mp', type=str, default='')
    parser.add_argument('--multi_env_cfg_name', '-mn', type=str, default='')
    parser.add_argument('--curriculum_cfg_path', '-cp', type=str, default='')
    parser.add_argument('--curriculum_cfg_name', '-cn', type=str, default='')
    parser.add_argument('--logdir', type=str, default='logs/ppo1')
    parser.add_argument('--cfg_env', type=str, default='cfg/ant.yaml')
    parser.add_argument('--cfg_train', type=str, default='cfg/train/ppo1.yaml')
    args = parser.parse_args()

    if args.multi_env_cfg_path:
        if args.multi_env_cfg_name == '':
            raise ValueError('Must give a name to the multi env cfg!')
        cfg_env = YamlConfig(os.path.join(args.multi_env_cfg_path, 
                        '{}_{}.yaml'.format(args.multi_env_cfg_name, hvd.rank())
                    ))
    else:
        cfg_env = YamlConfig(args.cfg_env)

    
    cfg_curr = None
    if args.curriculum_cfg_path:
        if args.curriculum_cfg_name == '':
            raise ValueError('Must give a name to the curriculum cfg!')
        cfg_curr = YamlConfig(os.path.join(args.curriculum_cfg_path, 
                    '{}_{}.yaml'.format(args.curriculum_cfg_name, hvd.rank())
                ))

    cfg_train = YamlConfig(args.cfg_train)

    train(args.logdir, cfg_env, cfg_train, args.no_distr_train, cfg_curr)