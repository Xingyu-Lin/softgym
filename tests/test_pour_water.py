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


args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='heuristic', help='heuristic or cem')
args.add_argument("--cem_traj_path", type=str, default='./data/traj/pour_water_cem_traj.pkl')
args.add_argument("--replay", type=int, default=0, help='if load pre-stored actions and make gifs')
args = args.parse_args()

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

if args.mode == 'heuristic':
    env = PourWaterPosControlEnv(observation_mode='cam_rgb', horizon=100, render=True, headless=False, num_variations=1000, 
                                 action_mode='direct', deterministic=True, render_mode='fluid')

    print("env make done!")
    timestep = env.horizon
    move_part = 15
    stable_part = int(0.0 * timestep)

    v = 0.08
    y = 0
    dt = 0.1
    x = 0
    total_rotate = 0.2 * np.pi

    # below is testing a naive heuristic policy
    print("total timestep: ", timestep)
    imgs = []
    for _ in range(1):
        env.reset()
        for i in range(50):
            if i < stable_part:
                action = np.array([0, 0, 0])

            elif stable_part <= i < move_part + stable_part:
                y = v * dt
                action = np.array([0.0012, y, 0.])

            elif i > move_part + stable_part and i < timestep - 30:
                theta = 1 / float(timestep - move_part - stable_part) * total_rotate
                action = np.array([0, 0, theta])

            else:
                action = np.array([0, 0, 0])
            # action = np.random.normal(scale=0.3, size=(1,3)).flatten()

            obs, reward, done, _ = env.step(action)

            img = env.render(mode='rgb_array')
            imgs.append(img)

            print("step {} reward {}".format(i, reward))
            if done:

                print("done!")
                break

    num = 8
    show_imgs = []
    factor = len(imgs) // num
    for i in range(num):
        img = imgs[i * factor].transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))
    
    grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs=grid_imgs[:, :, ::-1]
    cv2.imwrite('pour_water.jpg', grid_imgs)
    env.close()

elif args.mode == 'cem':
    from algorithms.cem import CEMPolicy
    import copy, pickle

    traj_path = args.cem_traj_path
    env = PourWaterPosControlEnv(observation_mode='full_state', action_mode='direct', horizon=75, deterministic=True,
                                 render_mode='fluid', headless=True, render=False)

    if not args.replay:
        policy = CEMPolicy(env,
                           plan_horizon=75,
                           max_iters=20,
                           population_size=50,
                           use_mpc=False,
                           num_elites=10)

        # Run policy
        obs = env.reset()
        initial_state = env.get_state()
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action(obs)
            action_traj.append(copy.deepcopy(action))
            obs, reward, _, _ = env.step(action)
            print('reward:', reward)

        traj_dict = {
            'initial_state': initial_state,
            'action_traj': action_traj
        }

        with open(traj_path, 'wb') as f:
            pickle.dump(traj_dict, f)
    else:
        with open(traj_path, 'rb') as f:
            traj_dict = pickle.load(f)
        initial_state, action_traj = traj_dict['initial_state'], traj_dict['action_traj']
        des_dir = './data/video/test_PourWater/'
        os.system('mkdir -p ' + des_dir)
        env.start_record(video_path=des_dir, video_name='cem_pour_water_2.gif')
        env.reset()
        env.set_state(initial_state)
        for action in action_traj:
            env.step(action)
        env.end_record()
