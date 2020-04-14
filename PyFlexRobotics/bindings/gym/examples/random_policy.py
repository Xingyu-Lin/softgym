import os, sys, argparse, logging
from collections import deque
from time import time

import numpy as np
from autolab_core import YamlConfig

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/ant.yaml')
    parser.add_argument('--no_logging', '-n', action='store_true')
    parser.add_argument('--zero_actions', '-z', action='store_true')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = make_flex_vec_env(cfg)
    logging.info('Total envs: {}'.format(env.num_envs))
    
    env.reset()
    frame_times = deque([0] * 10, 10)
    while True:
        # sample random actions from -1 to 1
        if args.zero_actions:
            actions = np.zeros((env.num_envs, env.num_acts))
        else:
            actions = np.random.rand(env.num_envs, env.num_acts) * 2 - 1

        s = time()
        _, rews, _, _ = env.step(actions)
        frame_times.append(time() - s)
        env.reset()

        if not args.no_logging:
            logging.info('mean reward: {:.3f} | avg frame time: {:.3f}ms'.format(
                np.mean(rews), np.mean(frame_times) * 1000)
            )
