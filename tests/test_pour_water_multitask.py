import gym
import numpy as np
import pyflex
from softgym.envs.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
from softgym.core.image_env import ImageEnv
from rlkit.envs.vae_wrapper import VAEWrappedEnv


args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--policy", type = str, default = 'heuristic', help = 'heuristic or cem')
args.add_argument("--cem_traj_path", type = str, default = '../data/traj/pour_water_cem_traj.pkl')
args.add_argument("--replay", type = int, default = 0, help = 'if load pre-stored actions and make gifs')
args = args.parse_args()


if args.policy == 'heuristic':
    env = PourWaterPosControlGoalConditionedEnv(observation_mode = 'full_state', horizon = 75, 
        action_mode = 'direct', deterministic=True, render_mode = 'fluid', render = True, headless= True)
    softgym.register_flex_envs()
    print("env make done")

    imsize = 128
    env = ImageEnv(
            env,
            imsize=imsize,
            transpose=True,
            normalize=True,
        )

    from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize84_default_architecture, imsize128_default_architecture
    vae_kwargs=dict(
                input_channels=3,
                architecture=imsize128_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            )

    from rlkit.pythonplusplus import identity
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
    )
    vae = ConvVAE(
        4,
        decoder_output_activation=identity,
        imsize=128,
        **vae_kwargs
    )

    env = VAEWrappedEnv(
                env,
                vae,
                imsize=env.imsize,
                decode_goals=False,
                render_goals=False,
                render_rollouts=False,
                )
            

    obs = env.reset()
    goal_img = obs['image_desired_goal'].reshape(-1, imsize, imsize).transpose()
    img_obs = obs['image_observation'].reshape(-1, imsize, imsize).transpose()

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(goal_img)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img_obs)
    plt.savefig('./debug.png')
    exit()

    # test that we have sampled a correct goal and that we have implemented the right set_to_goal
    # also to test we correctly constructed different goals 
    # for i in range(2):
    #     env.reset()
    #     env.set_to_goal(env.get_goal())
    #     img = env.get_image(960, 720)
    #     plt.imshow(img)
    #     plt.show()
    
    # exit()
    env.reset()

    timestep = env.horizon
    move_part = 15
    stable_part = int(0.0 * timestep)

    v = 0.13
    y = 0
    dt = 0.1
    x = env.glass_floor_centerx
    total_rotate = 0.28* np.pi

    # test a heuristic policy
    print("right before reset")
    env.reset()
    print("total timestep: ", timestep)
    for i in range(timestep):
        if i < stable_part:
            action = np.array([0, 0, 0])

        elif stable_part <= i < move_part + stable_part:
            y = v * dt
            action = np.array([0, y, 0.])

        elif i > move_part + stable_part and i < timestep - 30:
            theta = 1 / float(timestep - move_part - stable_part) * total_rotate
            action = np.array([0, 0, theta])

        else:
            action = np.array([0, 0 ,0])

        obs, reward, done, _ = env.step(action)

        # if i  == 250:
        #     # from matplotlib import pyplot as plt
        #     import cv2
        #     img = env.get_image(48, 48)
        #     cv2.imshow('test_img', img)
        #     cv2.waitKey(0)

        print("step {} reward {}".format(i, reward))
        if done:
            # env.end_record()
            
            print("done!")
            break

elif args.policy == 'cem':
    from algorithms.cem import CEMPolicy
    import copy, pickle

    traj_path = args.cem_traj_path
    env = PourWaterPosControlEnv(observation_mode = 'cam_rgb', action_mode = 'direct', horizon=300, deterministic=True,
        render_mode='fluid')

    if not args.replay:
        policy = CEMPolicy(env,
                        plan_horizon=100,
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
        env.start_record(video_path=des_dir, video_name='cem_pour_water_2.gif')
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