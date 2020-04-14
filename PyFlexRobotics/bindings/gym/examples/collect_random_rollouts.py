import os, sys, argparse, logging
from joblib import dump
from autolab_core import YamlConfig

from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
from flex_gym import RolloutCollector, RandomPolicy

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/ant.yaml')
    parser.add_argument('--filename', '-f', type=str, default='logs/random_rollouts.jb')
    parser.add_argument('--num_rollouts', '-n', type=int, default=100)
    args = parser.parse_args()

    cfg = YamlConfig(args.cfg)
    filename = os.path.realpath(args.filename)

    logging.info('Initializing env...')
    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = FlexVecEnv(cfg)

    logging.info('Collecting rollouts...')
    rollout_collector = RolloutCollector(env, RandomPolicy(env.num_acts))
    data = rollout_collector.get_rollouts(args.num_rollouts)
    env.close()

    logging.info('Saving to {}'.format(args.filename))

    dump(data, filename, compress=3)    
      