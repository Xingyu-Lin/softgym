import os, sys, argparse, logging
from collections import deque
from time import time
from copy import deepcopy

import numpy as np
from autolab_core import YamlConfig

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env_muli_env, FlexVecEnvMultiGPU

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    cfg = YamlConfig('cfg/humanoid_obstacles.yaml')
    
    # create 2 different cfgs
    cfgs = [deepcopy(cfg) for _ in range(2)]
    cfgs[0]['scene']['Density'] = 0
    cfgs[0]['gpus'] = [0]
    cfgs[1]['scene']['Density'] = 0.02
    cfgs[1]['gpus'] = [1]

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = make_flex_vec_env_muli_env(cfgs)
    logging.info('Total envs: {}'.format(env.num_envs))
    
    env.reset()
    frame_times = deque([0] * 10, 10)
    t = 0
    while True:
        if t < 100:
            t += 1
        # switching env
        if t % 100 == 0:
            t += 1
            logging.info('Switching envs')
            cfgs[0]['scene']['Density'] = 0.02
            cfgs[1]['scene']['Density'] = 0.04
            env.replace_env(0, cfgs[0])
            env.replace_env(1, cfgs[1])
            env.reset()

        # sample random actions from -1 to 1
        actions = np.random.rand(env.num_envs, env.num_acts) * 2 - 1

        s = time()
        _, rews, _, _ = env.step(actions)
        frame_times.append(time() - s)
        env.reset()

        logging.info('mean reward: {:.3f} | avg frame time: {:.3f}ms'.format(
            np.mean(rews), np.mean(frame_times) * 1000)
        )
