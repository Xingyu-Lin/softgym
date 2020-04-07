from softgym.envs.pass_water import PassWater1DEnv
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
args.add_argument("--headless", type=int, default=1)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args = args.parse_args()


def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])

def run_heuristic(args):
    mode = args.mode
    if mode == 'visual':
        env_name = "PassWaterGoal"
    else:
        env_name = "PassWater"
    print(args.headless)
    # exit()
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    if mode == 'visual':
        env = PassWater1DGoalConditionedEnv(**dic)
    else:
        env = PassWater1DEnv(**dic)


    returns = []
    final_performances = []
    N = 1 if mode == 'visual' else 100
    for _ in range(N):
        if mode != 'visual':
            env.eval_flag = True
            env.reset()
        else:
            config_id = 4
            env.reset(config_id=config_id)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=config_id)
            imgs = []

        total_reward = 0
        particle_y = get_particle_max_y()

        if np.abs(env.height - particle_y) > 0.2: # small water
            print("small")
        elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
            print("medium")
        else:
            print("large")

        horizon = env.horizon if mode != 'visual' else 35
        for i in range(horizon):
            if np.abs(env.height - particle_y) > 0.2: # small water
                action = np.array([0.13]) / action_repeat
            elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
                if mode != 'visual':
                    action = np.array([0.08]) / action_repeat
                else:
                    action = np.array([0.16]) / action_repeat
            else:
                action = np.array([0.025]) / action_repeat

            if np.abs(env.glass_x - env.terminal_x) < 0.1:
                action = np.array([0]) 
        
            _, reward, _, info = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render(mode='rgb_array'))

            if i == horizon - 1:
                final_performances.append(reward)

        print("episode total reward: ", total_reward)
        returns.append(total_reward)

    print("return mean: ", np.mean(returns))
    print("return std: ", np.std(returns))
    print("final performances mean {}".format(np.mean(final_performances)))

    if mode == 'visual':
        num = 7
        show_imgs = []
        factor = len(imgs) // num
        for i in range(num):
            img = imgs[i * factor].transpose(2, 0, 1)
            print(img.shape)
            show_imgs.append(torch.from_numpy(img.copy()))

        # goal_img = goal_img.transpose(2, 0, 1)
        # show_imgs.append(torch.from_numpy(goal_img.copy()))
        goal_img = goal_img[:, :, ::-1]
        save_name = 'data/icml/pass_water_goal.jpg'
        cv2.imwrite(save_name, goal_img)

        grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
        grid_imgs=grid_imgs[:, :, ::-1]
        save_path = 'data/icml/pass_water.jpg'
        print(save_path)
        cv2.imwrite(save_path, grid_imgs)

if __name__ == '__main__':
    run_heuristic(args)

# all_frames = imgs
# all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
# grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

# from os import path as osp
# save_name = 'pass_water_heuristic' + '.gif'
# save_numpy_as_gif(np.array(grid_imgs), osp.join('./data/video/env_demos', save_name))