import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.cloth_fold_multitask import ClothFoldGoalConditionedEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.normalized_env import normalize
import torch, torchvision, cv2, time
from softgym.registered_env import env_arg_dict
import argparse, sys

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='test')
args.add_argument("--headless", type=int, default=1)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args = args.parse_args()


def run_heuristic(args):
    mode = args.mode
    num_picker = 2
    env_name = 'ClothFold' if mode != 'visual' else 'ClothFoldGoal'
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic['action_repeat']
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = ClothFoldEnv(**dic) if mode != 'visual' else ClothFoldGoalConditionedEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    N = 1 if mode == 'visual' else 100
    for idx in range(N):
        if mode != 'visual':
            env.eval_flag = True
            env.reset()
        else:
            idx = 6
            env.reset(config_id=idx)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=idx)
        exit()
        total_reward = 0

        pos = pyflex.get_positions().reshape((-1, 4))
        minx = np.min(pos[:, 0])
        maxx = np.max(pos[:, 0])
        minz = np.min(pos[:, 2])
        maxz = np.max(pos[:, 2])

        corner1 = np.array([minx, 0.05, minz])
        corner2 = np.array([minx, 0.05, maxz])

        picker_pos, _ = env.action_tool._get_pos()

        differ1 = corner1 - picker_pos[0]
        differ2 = corner2 - picker_pos[1]

        steps = 15 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            action[0, :3] = differ1 / steps / action_repeat
            action[1, :3] = differ2 / steps / action_repeat

            obs, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        picker_pos, _ = env.action_tool._get_pos()

        steps = 15 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 1] = 0.002
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        pos = pyflex.get_positions().reshape((-1, 4))
        minx = np.min(pos[:, 0])
        maxx = np.max(pos[:, 0])
        minz = np.min(pos[:, 2])
        maxz = np.max(pos[:, 2])
        target_corner_1 = np.array([maxx, 0.10, minz])
        target_corner_2 = np.array([maxx, 0.10, maxz])
        picker_pos, _ = env.action_tool._get_pos()

        differ1 = target_corner_1 - picker_pos[0]
        differ2 = target_corner_2 - picker_pos[1]

        steps = 40 if mode != 'visual' else 30
        for i in range(steps):
            action = np.ones((num_picker, 4))
            action[0, :3] = differ1 / steps / action_repeat
            action[1, :3] = differ2 / steps / action_repeat
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        steps = 30 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))
            if i == steps - 1:
                final_performances.append(reward)

        print("episode {} total rewards {}".format(idx, total_reward))
        returns.append(total_reward)

    print("average return: ", np.mean(returns))
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
        save_name = 'data/icml/cloth_fold_goal.jpg'
        cv2.imwrite(save_name, goal_img)

        grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
        grid_imgs = grid_imgs[:, :, ::-1]
        save_name = 'data/icml/cloth_fold.jpg'
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
    test_random()
    # run_heuristic(args)
