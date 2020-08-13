import numpy as np
from softgym.envs.pour_water_amount import PourWaterAmountPosControlEnv
from softgym.multitask_envs_arxived.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
import argparse, sys
from matplotlib import pyplot as plt
from softgym.registered_env import  env_arg_dict

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='debug', help='visual: generate env images; otherwise, \
        run heuristic policy and evaluate its performance')
args.add_argument("--headless", type=int, default=0)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args = args.parse_args()

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def run_heuristic(args):
    mode = args.mode
    if mode == 'visual':
        env_name = 'PourWaterGoal'
    else:
        env_name = "PourWaterAmount"

    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    if args.mode == 'debug':
    #     dic['save_cached_states'] = False
    #     dic['use_cached_states'] = False
        dic['num_variations'] = 5

    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))

    if mode == 'visual':
        env = PourWaterPosControlGoalConditionedEnv(**dic)
    else:
        env = PourWaterAmountPosControlEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    if mode == 'visual':
        N = 1
    elif mode == 'debug':
        N = 5
    elif mode == 'test':
        N = 100
    else:
        N = 10

    goal_img = None
    for idx in range(N):
        total_reward = 0
        # env.eval_flag = True
        if mode == 'visual':
            env.reset(config_id=5)
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            env.reset(config_id=5)
        else:
            env.reset()
  
        print("target amount: ", env.current_config['target_amount'])
        move_part = 20
        target_y = env.poured_height + 0.2
        target_x = env.glass_distance - env.poured_glass_dis_x / 2 - env.height - 0.1
        for i in range(move_part):
            action = np.array([target_x / action_repeat / move_part , target_y / action_repeat / move_part, 0.])
            obs, reward, done, _ = env.step(action)
            print("reward: ", reward)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        
        rotate_part = 20
        total_rotate = 0.55 * np.pi
        for i in range(rotate_part):
            action = np.array([0.0005, 0.003, total_rotate / rotate_part / action_repeat])
            obs, reward, done, _ = env.step(action)
            print("reward: ", reward)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

        stay_part = 60 if mode != 'visual' else 21
        for i in range(stay_part):
            action = np.zeros(3)
            obs, reward, done, _ = env.step(action)
            print("reward: ", reward)
            total_reward += reward
            if mode == 'visual':
                imgs.append(env.render('rgb_array'))

            if i == stay_part - 1:
                final_performances.append(reward)

        returns.append(total_reward)
        print("episode {} total reward {}".format(idx, total_reward))

    env.close()
    return returns, final_performances, imgs, goal_img

if __name__ == '__main__':
    run_heuristic(args)

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
