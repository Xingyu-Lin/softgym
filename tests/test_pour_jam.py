import gym
import numpy as np
import pyflex
from softgym.envs.pour_jam import PourJamPosControlEnv
import os, argparse, sys

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--policy", type = str, default = 'heuristic', help = 'heuristic or cem')
args.add_argument("--cem_traj_path", type = str, default = '../data/traj/pour_water_cem_traj.pkl')
args.add_argument("--replay", type = int, default = 0, help = 'if load pre-stored actions and make gifs')
args = args.parse_args()

env = PourJamPosControlEnv(observation_mode = 'cam_rgb', action_mode = 'direct', deterministic=True)

timestep = env.horizon
move_part = int(0.3 * timestep)
stable_part = int(0.0 * timestep)

vy = 0.1
vx = 0.02
vx2 = 0.15
y = 0
dt = 0.1
x = env.glass_floor_centerx
total_rotate1 = 0.57* np.pi
total_rotate2 = 0.67* np.pi

move0 = 0.1 * timestep
move1 = 0.2 * timestep
rotate1 = 0.4 * timestep
rotateback1 = 0.55 * timestep
move2 = 0.65 * timestep
rotate2 = 0.85 * timestep
rotateback2 = 0.95 * timestep

# env.start_record(video_path='../data/video/', video_name='pour_jam_fluid.gif')
env.reset()
if args.policy == 'heuristic':
    for i in range(timestep):
        if i < move0:
            y = y + vy * dt
            action = np.array([x, y, 0.])

        elif i >= move0 and i < move1:
            x = x + vx * dt
            action = np.array([x, y, 0])
        
        elif i >= move1 and i < rotate1:
            theta = (i - move1) / float(rotate1 - move1) * total_rotate1
            action = np.array([x, y, theta])

        elif i >= rotate1 and i < rotateback1:
            theta = (1 - (i - rotate1) / float(rotateback1 - rotate1)) * total_rotate1
            action = np.array([x, y, theta])

        elif i >= rotateback1 and i < move2:
            x = x + vx2 * dt
            action = np.array([x, y, theta])
        
        elif i >= move2 and i < rotate2:
            theta = (i - move2) / float(rotate2 - move2) * total_rotate2
            action = np.array([x, y, theta])

        elif i >= rotate2 and i < rotateback2:
            theta = (1 - (i - rotate2) / float(rotateback2 - rotate2)) * total_rotate2
            action = np.array([x, y, theta])
        

        _, reward, done, _ = env.step(action)

        # print("step {} reward {}".format(i, reward))
        if done:
            # env.end_record()
            break

# elif args.policy == 'cem':
#     from algorithms.cem import CEMPolicy
#     import copy, pickle

#     traj_path = args.cem_traj_path

#     if not args.replay:
#         policy = CEMPolicy(env,
#                         plan_horizon=30,
#                         max_iters=10,
#                         population_size=30,
#                         num_elites=5)
#         # Run policy
#         obs = env.reset()
#         initial_state = env.get_state()
#         action_traj = []
#         for _ in range(env.horizon):
#             action = policy.get_action(obs)
#             env.debug = True
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
#         des_dir = '../data/video/test_PourWater/'
#         os.system('mkdir -p ' + des_dir)    
#         env.start_record(video_path=des_dir, video_name='cem_pour_water.gif')
#         env.reset()
#         env.set_state(initial_state)
#         for action in action_traj:
#             env.step(action)
#         env.end_record()

    

# env.reset()
# for i in range(timestep):
#     pyflex.set_positions(positions[i])
#     pyflex.set_shape_states(shape_states[i, :-1]) ### render removes front wall

#     pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

# pyflex.clean()