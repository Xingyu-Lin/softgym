import gym
import numpy as np
import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import os, argparse, sys

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--policy", type = str, default = 'heuristic', help = 'heuristic or cem')
args.add_argument("--cem_traj_path", type = str, default = '../data/traj/pour_water_cem_traj.pkl')
args.add_argument("--replay", type = int, default = 0, help = 'if load pre-stored actions and make gifs')
args = args.parse_args()


if args.policy == 'heuristic':
    env = PourWaterPosControlEnv(observation_mode = 'cam_img', horizon = 300, 
        action_mode = 'direct', deterministic=True, render_mode = 'fluid')

    timestep = env.horizon
    move_part = 50
    stable_part = int(0.0 * timestep)

    v = 0.1
    y = 0
    dt = 0.1
    x = env.glass_floor_centerx
    total_rotate = 0.7* np.pi

    # env.start_record(video_path='../data/video/', video_name='pour_water.gif')
    env.reset()
    for i in range(timestep):
        if i < stable_part:
            action = np.array([0, 0, 0])

        elif stable_part <= i < move_part + stable_part:
            # if i % 2 == 0:
            #     x = v * dt
            # else:
            #     x = -v * dt
            y = v * dt
            action = np.array([0, y, 0.])

        elif i > move_part + stable_part and i < timestep - 50:
            # theta = min(1, (i - move_part - stable_part) / float(timestep - 200 - move_part - stable_part)) * total_rotate
            theta = 1 / float(timestep -50 - move_part - stable_part) * total_rotate
            # print(theta / np.pi)
            action = np.array([0, 0, theta])

        _, reward, done, _ = env.step(action)

        print("step {} reward {}".format(i, reward))
        if done:
            # env.end_record()
            break

elif args.policy == 'cem':
    from algorithms.cem import CEMPolicy
    import copy, pickle

    traj_path = args.cem_traj_path
    env = PourWaterPosControlEnv(observation_mode = 'cam_img', action_mode = 'direct', horizon=150, deterministic=True)

    if not args.replay:
        policy = CEMPolicy(env,
                        plan_horizon=50,
                        max_iters=5,
                        population_size=100,
                        num_elites=10)
        # Run policy
        obs = env.reset()
        initial_state = env.get_state()
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action(obs)
            env.debug = True
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
        des_dir = '../data/video/test_PourWater/'
        os.system('mkdir -p ' + des_dir)    
        env.start_record(video_path=des_dir, video_name='cem_pour_water_1.gif')
        env.reset()
        env.set_state(initial_state)
        for action in action_traj:
            env.step(action)
        env.end_record()

    

# env.reset()
# for i in range(timestep):
#     pyflex.set_positions(positions[i])
#     pyflex.set_shape_states(shape_states[i, :-1]) ### render removes front wall

#     pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

# pyflex.clean()