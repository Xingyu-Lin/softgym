import gym
import numpy as np
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_manipulate import ClothManipulateEnv
from softgym.utils.visualization import save_numpy_as_gif
# from softgym.utils.visualization import save_numpy_to_gif_matplotlib
import torch, torchvision, cv2, time
from softgym.registered_env import  env_arg_dict
import argparse, sys

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='heuristic', help='heuristic or cem')
args = args.parse_args()

def run_heuristic(mode='visual'):
    num_picker = 2
    env_name = 'ClothFlatten' if mode != 'visual' else 'ClothManipulate'
    dic = env_arg_dict[env_name]
    dic['headless'] = True
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = ClothFlattenEnv(**dic) if mode != 'visual' else ClothManipulateEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    N = 1 if mode == 'visual' else 100
    for idx in range(N):
        if mode != 'visual':
            env.eval_flag = True
            env.reset()
        else:
            idx = 14
            env.reset(config_id=idx)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=idx)

        total_reward = 0

        steps = 5
        for i in range(steps):
            action = np.zeros((2, 4))
            action[:, 1] = 0.002
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        corner_idx = env._get_key_point_idx()

        first_pos = pyflex.get_positions().reshape((-1, 4))[corner_idx[0], :3]
        last_pos = pyflex.get_positions().reshape((-1, 4))[corner_idx[1], :3]
        first_pos1 = pyflex.get_positions().reshape((-1, 4))[corner_idx[2], :3]
        last_pos1 = pyflex.get_positions().reshape((-1, 4))[corner_idx[3], :3]
        mean_z = (first_pos[2] + last_pos[2]) / 2
        mean_z1 = (first_pos1[2] + last_pos1[2]) / 2
        if mean_z > mean_z1:
            move = 0.002
        else:
            move = -0.002

        picker_pos, _ = env.action_tool._get_pos()
        if mode != 'visual':
            diff_first = first_pos - picker_pos[0]
            diff_last = last_pos - picker_pos[1]
        else:
            diff_first = last_pos1 - picker_pos[0]
            diff_last = last_pos - picker_pos[1]

        move_step = 20
        for i in range(move_step):
            action = np.zeros((num_picker, 4))
            action[0, :3] = diff_first / move_step / env.action_repeat
            action[1, :3] = diff_last / move_step / env.action_repeat
            _ ,reward,_, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        target_1 = np.array([-1.2, 0.05, -1.2])
        target_2 = np.array([-0.5, 0.05, -1.5])
        picker_pos, _ = env.action_tool._get_pos()
        diff_first = target_1 - picker_pos[0]
        diff_last = target_2 - picker_pos[1]

        move_step = 40
        for i in range(move_step):
            if mode == 'visual':
                action = np.zeros((num_picker, 4))
                action[0, :3] = diff_first / move_step / env.action_repeat
                action[1, :3] = diff_last / move_step / env.action_repeat
                action[:, -1] = 1
                
            else:
                action[:, 2] = move
                action[:, -1] = 1
                # # action[0, :3] = 0
            _ ,reward,_, _ = env.step(action)
            total_reward += reward

            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        move_step = 35 if mode != 'visual' else 5
        for i in range(move_step):
            action = np.zeros((2, 4))
            _ ,reward, _, _ = env.step(action)
            total_reward += reward 
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))
            if i == move_step - 1:
                final_performances.append(reward)

        print("episode {} total return {}".format(idx, total_reward))
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
        save_name = 'data/icml/cloth_flatten_goal.jpg'
        cv2.imwrite(save_name, goal_img)

        grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
        grid_imgs=grid_imgs[:, :, ::-1]
        save_name = 'data/icml/cloth_flatten.jpg'
        print(save_name)
        cv2.imwrite(save_name, grid_imgs)

def test_random(env, N=5):
    N = 5
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    run_heuristic(args.mode)