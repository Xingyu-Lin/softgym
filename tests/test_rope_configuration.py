import gym
import numpy as np
import pyflex
from softgym.envs.rope_configuration import RopeConfigurationEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
import torch, torchvision, cv2
from softgym.registered_env import  env_arg_dict
import argparse

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='debug')
args.add_argument("--headless", type=int, default=1)
args.add_argument("--N", type=int, default=8)
args.add_argument("--use_cached_states", type=bool, default=False)
args.add_argument("--obs_mode", type=str, default='key_point')
args = args.parse_args()

def run_heuristic(args):
    env_name = 'RopeConfiguration'
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode

    if not args.use_cached_states:
        dic['save_cached_states'] = True
        dic['use_cached_states'] = False
        dic['num_variations'] = 10


    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = RopeConfigurationEnv(**dic) 

    N = args.N
    imgs = []
    returns = []
    final_performances = []
    for idx in range(N):
        obs = env.reset()
        total_reward = 0
        
        pos = pyflex.get_positions().reshape((-1, 4))
        corner1 = pos[0][:3]
        corner2 = pos[-1][:3]

        steps = 5
        for i in range(steps):
            action = np.zeros((2, 4))
            action[:, 1] = 0.01
            obs, reward, _, _ = env.step(action)
            total_reward += reward
            imgs.append(env.render('rgb_array'))

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = corner1 - picker_pos[0]
        diff2 = corner2 - picker_pos[1]

        steps = 15 
        for i in range(steps):
            action = np.zeros((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            _, reward, _, _ = env.step(action)
            total_reward += reward
            imgs.append(env.render('rgb_array'))


        picker_pos, _ = env.action_tool._get_pos()
        target_pos_1 = np.array([2.7, 0.05, 0])
        target_pos_2 = np.array([-2.7, 0.05, 0])

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = target_pos_1 - picker_pos[0]
        diff2 = target_pos_2 - picker_pos[1]

        steps = 20
        for i in range(steps):
            action = np.ones((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            _, reward, _ ,_  = env.step(action)
            total_reward += reward
            imgs.append(env.render('rgb_array'))

        steps = 35 
        for i in range(steps):
            action = np.zeros((2, 4))
            _, reward, _ ,_  = env.step(action)
            total_reward += reward
            imgs.append(env.render('rgb_array'))
            if i == steps - 1:
                final_performances.append(reward)

        print("episode {} total reward {}".format(idx, total_reward))
        returns.append(total_reward)

    print("mean return: ", np.mean(returns))
    print("std return: ", np.std(returns))
    print("final performances mean {}".format(np.mean(final_performances)))

if __name__ == '__main__':
    run_heuristic(args)