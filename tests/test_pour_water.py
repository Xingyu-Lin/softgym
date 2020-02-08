import gym
import numpy as np
import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.envs.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
from softgym.utils.visualization import save_numpy_as_gif
import torchvision
import torch
import cv2
from matplotlib import pyplot as plt
from softgym.registered_env import  env_arg_dict


args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='heuristic', help='heuristic or cem')
args = args.parse_args()

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def run_heuristic(mode='test'):
    if mode == 'visual':
        env_name = 'PourWaterGoal'
    else:
        env_name = "PourWater"

    dic = env_arg_dict[env_name]
    dic['headless'] = True
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))

    if mode == 'visual':
        env = PourWaterPosControlGoalConditionedEnv(**dic)
    else:
        env = PourWaterPosControlEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    if mode == 'visual':
        N = 1
    elif mode == 'test':
        N = 100
    for idx in range(N):
        total_reward = 0
        env.eval_flag = True
        if mode == 'visual':
            env.reset(config_id=5)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=5)
        else:
            env.reset()
  
        move_part = 20
        target_y = env.poured_height + 0.2
        target_x = env.glass_distance - env.poured_glass_dis_x / 2 - env.height - 0.1
        for i in range(move_part):
            action = np.array([target_x / action_repeat / move_part , target_y / action_repeat / move_part, 0.])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        
        rotate_part = 20
        total_rotate = 0.55 * np.pi
        for i in range(rotate_part):
            action = np.array([0.0005, 0.003, total_rotate / rotate_part / action_repeat])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        stay_part = 60 if mode != 'visual' else 21
        for i in range(stay_part):
            action = np.zeros(3)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

            if i == stay_part - 1:
                final_performances.append(reward)

        returns.append(total_reward)
        print("episode {} total reward {}".format(idx, total_reward))


    print("returns mean {}".format(np.mean(returns)))
    print("returns std {}".format(np.std(returns)))
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
        save_name = 'data/icml/pour_water_goal.jpg'
        cv2.imwrite(save_name, goal_img)

        grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
        grid_imgs=grid_imgs[:, :, ::-1]
        print('data/icml/pour_water.jpg')
        cv2.imwrite('data/icml/pour_water.jpg', grid_imgs)
    env.close()


if __name__ == '__main__':
    run_heuristic(args.mode)

# elif args.mode == 'cem':
#     from algorithms.cem import CEMPolicy
#     import copy, pickle

#     traj_path = args.cem_traj_path
#     env = PourWaterPosControlEnv(observation_mode='full_state', action_mode='direct', horizon=75, deterministic=True,
#                                  render_mode='fluid', headless=True, render=False)

#     if not args.replay:
#         policy = CEMPolicy(env,
#                            plan_horizon=75,
#                            max_iters=20,
#                            population_size=50,
#                            use_mpc=False,
#                            num_elites=10)

#         # Run policy
#         obs = env.reset()
#         initial_state = env.get_state()
#         action_traj = []
#         for _ in range(env.horizon):
#             action = policy.get_action(obs)
#             action_traj.append(copy.deepcopy(action))
#             obs, reward, _, _ = env.step(action)
#             print('reward:', reward)

#         traj_dict = {
#             'initial_state': initial_state,
#             'action_traj': action_traj
#         }

#         with open(traj_path, 'wb') as f:
#             pickle.dump(traj_dict, f)
#     else:
#         with open(traj_path, 'rb') as f:
#             traj_dict = pickle.load(f)
#         initial_state, action_traj = traj_dict['initial_state'], traj_dict['action_traj']
#         des_dir = './data/video/test_PourWater/'
#         os.system('mkdir -p ' + des_dir)
#         env.start_record(video_path=des_dir, video_name='cem_pour_water_2.gif')
#         env.reset()
#         env.set_state(initial_state)
#         for action in action_traj:
#             env.step(action)
#         env.end_record()
