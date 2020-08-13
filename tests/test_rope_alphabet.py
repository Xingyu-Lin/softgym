import numpy as np
import pyflex
from softgym.envs.rope_alphabet import RopeAlphaBetEnv
from softgym.multitask_envs_arxived.rope_manipulate import RopeManipulateEnv
import sys
from matplotlib import pyplot as plt
import torch, torchvision, cv2
from softgym.registered_env import  env_arg_dict
import argparse

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='debug')
args.add_argument("--headless", type=int, default=1)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args = args.parse_args()

def run_heuristic(args):
    mode = args.mode
    env_name = 'RopeAlphaBet' if mode != 'visual' else "RopeManipulate"
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode

    if mode == 'debug':
        # dic['use_cached_states'] = False
        # dic['save_cached_states'] = False
        dic['num_variations'] = 20

    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = RopeAlphaBetEnv(**dic) if mode != 'visual' else RopeManipulateEnv(**dic)

    N = 100 if mode != 'visual' else 1
    imgs = []
    returns = []
    final_performances = []
    for idx in range(N):
        if mode != 'visual':
            # env.eval_flag = True
            obs = env.reset()
            print(obs.shape)
            current_image = obs[:, :, :3]
            goal_image = obs[:, :, 3:]
            new_image = np.concatenate([current_image, goal_image], axis=1)
            plt.imshow(new_image)
            plt.show()
            # exit()
        else:
            idx = 6
            env.reset(config_id=idx)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=idx)

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
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = corner1 - picker_pos[0]
        diff2 = corner2 - picker_pos[1]

        steps = 15 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
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
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        steps = 35 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((2, 4))
            _, reward, _ ,_  = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))
            if i == steps - 1:
                final_performances.append(reward)

        print("episode {} total reward {}".format(idx, total_reward))
        returns.append(total_reward)

    print("mean return: ", np.mean(returns))
    print("std return: ", np.std(returns))
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
        save_name = 'data/icml/rope_flatten_goal.jpg'
        cv2.imwrite(save_name, goal_img)

        grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
        grid_imgs=grid_imgs[:, :, ::-1]
        save_name = 'data/icml/rope_flatten.jpg'
        print(save_name)
        cv2.imwrite(save_name, grid_imgs)

if __name__ == '__main__':
    run_heuristic(args)