import gym
import numpy as np
import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
from softgym.utils.visualization import save_numpy_as_gif
import torchvision
import torch
import cv2
from matplotlib import pyplot as plt
from softgym.registered_env import  env_arg_dict
import os.path as osp


args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='heuristic', help='visual: generate env images; otherwise, \
        run heuristic policy and evaluate its performance')
args.add_argument("--headless", type=int, default=0)
args.add_argument("--use_cached_states", type=bool, default=False)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args.add_argument("--N", type=int, default=5)
args = args.parse_args()

def run_heuristic(args):
    env_name = "PourWater"

    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    # dic['observation_mode'] = 'key_point'
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    
    # if not args.use_cached_states:
    #     dic['save_cache_states'] = True
    #     dic['use_cached_states'] = False
    #     dic['num_variations'] = 5

    dic['render_mode'] = 'fluid'

    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = PourWaterPosControlEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    N = args.N

    for idx in range(N):
        total_reward = 0
        env.reset()
        img = env.get_image(720, 720)
        cv2.imshow('name', img)
        cv2.waitKey()

        move_part = 15
        target_y = env.poured_height + 0.01
        target_x = env.glass_distance - env.poured_glass_dis_x / 2 - env.height - 0.05
        for i in range(move_part):
            action = np.array([target_x / action_repeat / move_part , target_y / action_repeat / move_part, 0.])
            obs, reward, done, info = env.step(action)
            total_reward += reward
            img = env.get_image(84, 84)
            cv2.imshow('name', img)
            cv2.waitKey()
            imgs.append(env.render('rgb_array'))
            print(reward, info['performance'])

        
        rotate_part = 20
        total_rotate = 0.55 * np.pi
        for i in range(rotate_part):
            action = np.array([0.0002, 0.002, total_rotate / rotate_part / action_repeat])
            obs, reward, done, info = env.step(action)
            total_reward += reward
            imgs.append(env.render('rgb_array'))
            print(reward, info['performance'])


        stay_part = 20
        for i in range(stay_part):
            action = np.zeros(3)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            imgs.append(env.render('rgb_array'))

            if i == stay_part - 1:
                final_performances.append(reward)

            print(reward, info['performance'])
            
        returns.append(total_reward)
        print("episode {} total reward {}".format(idx, total_reward))

    env.close()
    return returns, final_performances, imgs

if __name__ == '__main__':
    run_heuristic(args)
