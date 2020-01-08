import gym
import numpy as np
import pyflex
import os.path as osp
from softgym.envs.cloth_flatten import ClothFlattenEnv
import os, argparse, sys
from algorithms.cem import CEMPolicy
import copy, pickle

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--policy", type=str, default='heuristic', help='heuristic or cem')
args.add_argument("--cem_traj_path", type=str, default='./data/traj/pour_water_cem_traj.pkl')
args.add_argument("--replay", type=int, default=0, help='if load pre-stored actions and make gifs')
args = args.parse_args()

if __name__ == '__main__':
    traj_path = args.cem_traj_path
    cur_dir = osp.dirname(osp.abspath(__file__))
    cached_file = osp.join(cur_dir, 'cem_cloth_flatten_init_states.pkl')
    env = ClothFlattenEnv(
        cached_init_state_path=cached_file,
        observation_mode='key_point',
        action_mode='picker',
        num_picker=4,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth')
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
        des_dir = './data/video/cem_flatten'
        os.system('mkdir -p ' + des_dir)
        env.start_record()
        env.reset()
        env.set_state(initial_state)
        for action in action_traj:
            env.step(action)
        env.end_record(video_path=os.path.join(des_dir, 'cem_flatten.gif'))
