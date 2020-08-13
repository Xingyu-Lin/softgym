import numpy as np
import pyflex
from softgym.envs.rope_flatten_new import RopeFlattenNewEnv
from softgym.multitask_envs_arxived.rope_manipulate import RopeManipulateEnv
import sys
from softgym.registered_env import  env_arg_dict
import argparse
import time
from softgym.utils.visualization import save_numpy_as_gif

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='debug')
args.add_argument("--headless", type=int, default=0)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args.add_argument("--use_cached_states", type=str, default=False)
args.add_argument("--N", type=int, default=1)
args = args.parse_args()

def run_heuristic(args):
    mode = args.mode
    env_name = 'RopeFlattenNew' if mode != 'visual' else "RopeManipulate"
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode

    # if not args.use_cached_states:
    #     dic['use_cached_states'] = False
    #     dic['save_cache_states'] = True
    #     dic['num_variations'] = 5

    action_repeat = 1
    dic['action_repeat'] = action_repeat
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = RopeFlattenNewEnv(**dic) if mode != 'visual' else RopeManipulateEnv(**dic)

    N = args.N
    imgs = []
    returns = []
    final_performances = []
    for idx in range(N):
        env.reset()
        rope_len = env.rope_length
        print("rope segment: ", env.current_config['segment'])
        print("theoretic rope length: ", env.current_config['segment'] * env.current_config['radius'] * 0.5)
        print("rope initial len: ", rope_len)
        print("current end point distance: ", env._get_endpoint_distance())
        
        time.sleep(2)

        # env.action_tool.visualize_picker_boundary()

        total_reward = 0
        
        pos = pyflex.get_positions().reshape((-1, 4))
        corner1 = pos[4][:3]
        corner2 = pos[-1][:3]

        print("moving up picker!")
        time.sleep(1)
        steps = 5
        for i in range(steps):
            action = np.zeros((2, 4))
            action[:, 1] = 0.01
            obs, reward, _, info = env.step(action)
            total_reward += reward
            # print(env.action_tool._get_pos()[0])
            # time.sleep(0.1)
            print("obs: ", obs)
            # print(reward, info['performance'])
            imgs.append(env.get_image(720, 720))

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = corner1 - picker_pos[0]
        diff2 = corner2 - picker_pos[1]

        print("moving picker to corner!")

        time.sleep(1)
        steps = 50
        for i in range(steps):
            action = np.zeros((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            obs, reward, _, info = env.step(action)
            total_reward += reward
            print("obs:", obs)

            # print(env.action_tool._get_pos()[0])
            # time.sleep(0.1)
            # print(reward, info['performance'])
            imgs.append(env.get_image(720, 720))


        target_pos_1 = np.array([rope_len / 2, 0.05, 0])
        target_pos_2 = np.array([-rope_len / 2, 0.05, 0])

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = target_pos_1 - picker_pos[0]
        diff2 = target_pos_2 - picker_pos[1]

        # print("moving picker to target location!")
        # print("picker pos 0: ", picker_pos[0])
        # print("picker pos 1: ", picker_pos[1])
        # print("difference 1: ", diff1)
        # print("difference 2: ", diff2)
        time.sleep(1)
        steps = 50
        for i in range(steps):
            action = np.ones((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            # print(action)
            obs, reward, _ , info  = env.step(action)
            print("obs: ", obs)
            total_reward += reward
            # print(reward, info['performance'])
            time.sleep(0.2)
            # print(env.action_tool._get_pos()[0])
            imgs.append(env.get_image(720, 720))

        steps = 10
        for i in range(steps):
            action = np.zeros((2, 4))
            _, reward, _ , info  = env.step(action)
            total_reward += reward
            if i == steps - 1:
                final_performances.append(reward)
            time.sleep(0.1)
            # print(reward, info['performance'])
            imgs.append(env.get_image(720, 720))


        print("episode {} total reward {}".format(idx, total_reward))
        returns.append(total_reward)

    save_dir = 'data/videos/'
    save_name = 'RopeFlatten'
    save_numpy_as_gif(np.asarray(imgs), save_dir + save_name)
    print("mean return: ", np.mean(returns))
    print("std return: ", np.std(returns))
    print("final performances mean {}".format(np.mean(final_performances)))

if __name__ == '__main__':
    run_heuristic(args)