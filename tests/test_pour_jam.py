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

env = PourJamPosControlEnv(observation_mode = 'cam_img', action_mode = 'direct')

timestep = env.horizon
move_part = int(0.3 * timestep)
stable_part = int(0.0 * timestep)

v = 0.1
y = 0
dt = 0.1
x = env.glass_floor_centerx
total_rotate = 0.6* np.pi
dim_position = 4
dim_velocity = 3
dim_shape_state = 14

n_particles = pyflex.get_n_particles()
n_shapes = pyflex.get_n_shapes()
n_rigids = pyflex.get_n_rigids()
n_rigidPositions = pyflex.get_n_rigidPositions()
positions = np.zeros((timestep, n_particles, dim_position))
velocities = np.zeros((timestep, n_particles, dim_velocity))
shape_states = np.zeros((timestep, n_shapes, dim_shape_state))

env.reset()
if args.policy == 'heuristic':
    for i in range(timestep):
        if i < stable_part:
            action = np.array([x, y, 0])

        elif stable_part <= i < move_part + stable_part:
            y = y + v * dt
            action = np.array([x, y, 0.])

        else:
            theta = min(1, (i - move_part - stable_part) / float(timestep - 100 - move_part - stable_part)) * total_rotate
            # print(theta / np.pi)
            action = np.array([x, y, theta])

        positions[i] = pyflex.get_positions().reshape(-1, dim_position)
        velocities[i] = pyflex.get_velocities().reshape(-1, dim_velocity)
        shape_states[i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)   
        _, reward, done, _ = env.step(action)

        # print("step {} reward {}".format(i, reward))
        if done:
            break

elif args.policy == 'cem':
    from algorithms.cem import CEMPolicy
    import copy, pickle

    traj_path = args.cem_traj_path

    if not args.replay:
        policy = CEMPolicy(env,
                        plan_horizon=30,
                        max_iters=10,
                        population_size=30,
                        num_elites=5)
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
        env.start_record(video_path=des_dir, video_name='cem_pour_water.gif')
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