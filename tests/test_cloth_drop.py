import numpy as np
from softgym.envs.cloth_drop import ClothDropEnv
from softgym.multitask_envs_arxived.cloth_drop_multitask import ClothDropGoalConditionedEnv
import sys
import torch, torchvision, cv2
from softgym.registered_env import  env_arg_dict
import argparse

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='heuristic', help='heuristic or cem')
args = args.parse_args()


def run_heuristic(mode = 'visual'):
    num_picker = 2
    env_name = "ClothDropGoal" if mode == 'visual' else 'ClothDrop' 
    dic = env_arg_dict[env_name]
    dic['headless'] = True
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = ClothDropEnv(**dic) if mode != 'visual' else ClothDropGoalConditionedEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    N = 100 if mode != 'visual' else 1
    for _ in range(N):
        total_reward = 0
        if mode != 'visual':
            env.eval_flag = True
            env.reset()
        else:
            idx = 15
            env.reset(config_id=idx)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=idx)

        fast_move_steps = 5
        for i in range(fast_move_steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 0] = 0.5 / env.action_repeat 
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        slow_move_steps = 3
        for i in range(slow_move_steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 0] = -0.6 / env.action_repeat
            action[:, 1] = -0.12 / env.action_repeat
            _, reward, _, _ =env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        let_go_steps = 6
        for i in range(let_go_steps):
            action = np.zeros((num_picker, 4))
            _, reward, _, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))
            if i == let_go_steps - 1:
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
        save_name = 'data/icml/cloth_drop_goal.jpg'
        cv2.imwrite(save_name, goal_img)

        grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
        grid_imgs=grid_imgs[:, :, ::-1]
        save_name = 'data/icml/cloth_drop.jpg'
        print(save_name)
        cv2.imwrite(save_name, grid_imgs)


def test_random(env, N=5):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            action = action.reshape(-1, 4)
            _, reward, _, _ = env.step(action.flatten())
            print(reward)


if __name__ == '__main__':
    run_heuristic(args.mode)
