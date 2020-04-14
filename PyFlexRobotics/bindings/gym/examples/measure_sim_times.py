import os, sys, argparse, logging
from time import time

import numpy as np
import pandas as pd
from autolab_core import YamlConfig

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env
import GPUtil

S_TO_MS = 1000

def get_frame_times(cfg, n, T):
    cfg['scene']['NumAgents'] = n
    env = make_flex_vec_env(cfg)
    
    env.reset()
    frame_times = []
    sim_times = []
    for _ in range(T):
        # sample random actions from -1 to 1
        actions = np.random.rand(env.num_envs, env.num_acts) * 2 - 1

        s = time()
        env.step(actions, times=sim_times)
        frame_times.append(time() - s)
        env.reset()

    gpu = GPUtil.getGPUs()[0]
    gpu_util = gpu.load
    gpu_used_mem = gpu.memoryUsed

    env.close()

    frame_times = np.array(frame_times) * S_TO_MS
    sim_times = np.array(sim_times) * S_TO_MS
    io_times = frame_times - sim_times

    return {
        'num_agents': n,
        'frame_time_mean': np.mean(frame_times),
        'frame_time_med': np.median(frame_times),
        'frame_time_std': np.std(frame_times),
        'sim_time_mean': np.mean(sim_times),
        'sim_time_med': np.median(sim_times),
        'sim_time_std': np.std(sim_times),
        'io_time_mean': np.mean(io_times),
        'io_time_med': np.median(io_times),
        'io_time_std': np.std(io_times),
        'gpu_util': gpu_util,
        'gpu_mem': gpu_used_mem
    }


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/ant.yaml')
    parser.add_argument('--samples', '-T', type=int, default=1000)
    parser.add_argument('--num_agents_start', '-s', type=int, default=0)
    parser.add_argument('--num_agents_end', '-n', type=int, default=2000)
    parser.add_argument('--incre', '-i', type=int, default=5)
    parser.add_argument('--logdir', '-l', type=str)
    args = parser.parse_args()
    
    logdir = os.path.realpath(args.logdir)
    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))

    cfg = YamlConfig(args.cfg)
    cfg['exp']['frame_sampling']['mean'] = 1
    cfg['exp']['max_ep_len'] = args.samples

    records = []
    ns = list(range(args.num_agents_start, args.num_agents_end, args.incre))
    if ns[0] == 0:
        ns[0] = 1
    for n in ns:
        logging.info('Evaluating {} agents'.format(n))
        records.append(get_frame_times(cfg, n, args.samples))

        data = pd.DataFrame(records)
        data.to_csv(logdir, index=False)