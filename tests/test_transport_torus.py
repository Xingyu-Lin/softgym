from softgym.envs.transport_torus import TransportTorus1D
from softgym.envs.pass_water_multitask import PassWater1DGoalConditionedEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym
from softgym.utils.visualization import save_numpy_as_gif
import time
import torchvision, torch
import cv2
from softgym.registered_env import  env_arg_dict
import argparse, sys

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='test')
args.add_argument("--headless", type=int, default=0)
args.add_argument("--N", type=int, default=1)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args.add_argument("--use_cached_states", type=str, default=False)
args = args.parse_args()
from softgym.utils.visualization import save_numpy_as_gif

def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])

def run_heuristic(args):
    env_name = "PassWaterTorus"
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    dic['render_mode'] = 'particle'
    action_repeat = 8
    dic['action_repeat'] = action_repeat
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))

    # if not args.use_cached_states:
    #     dic['use_cached_states'] = False
    #     dic['save_cache_states'] = True
    #     dic['num_variations'] = 5

    env = TransportTorus1D(**dic)

    returns = []
    final_performances = []
    imgs = []
    N = args.N
    for _ in range(N):
        env.reset()
        print(env.action_space.high)
        print(env.action_space.low)
        print(env.terminal_x)
        print(env.box_params['box_dis_x'])
        print(env.box_params['box_dis_z'])
        total_reward = 0

        horizon = env.horizon 
        for i in range(horizon):

            action = np.array([0.007])
            # action = env.action_space.sample()

            if np.abs(env.box_x - env.terminal_x) < 0.05:
                action = np.array([0]) 

        
            _, reward, _, info = env.step(action)
            # time.sleep(0.2)
            total_reward += reward
            imgs.append(env.get_image(width=128, height=128))
            print(reward, info['performance'])

            if i == horizon - 1:
                final_performances.append(reward)

        print("episode total reward: ", total_reward)
        returns.append(total_reward)

    print("return mean: ", np.mean(returns))
    print("return std: ", np.std(returns))
    print("final performances mean {}".format(np.mean(final_performances)))
    save_dir = './data/videos/'
    save_name = 'TransportTorus'
    save_numpy_as_gif(np.array(imgs), save_dir + save_name, fps=5)

if __name__ == '__main__':
    run_heuristic(args)