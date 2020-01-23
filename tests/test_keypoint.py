import gym
import numpy as np
import pyflex
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.pour_water import PourWaterPosControlEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt


def get_action_towards_key_point(obs, num_picker):
    obs = obs.reshape([-1, 3])
    action = np.zeros([num_picker, 4], dtype=np.float32)
    for i in range(num_picker):
        action[i, :3] = (obs[i, :3] - obs[i + num_picker, :3]) / 160.
    return action


def test_keypoint(env, num_picker):
    for _ in range(10):
        obs = env.reset()
        action = get_action_towards_key_point(obs, num_picker)
        for i in range(20):
            print('time:', i)
            obs, _, _, _ = env.step(action.flatten())


def test_random(env, N=5):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        obs = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            print(obs)


if __name__ == '__main__':
    num_picker = 2
    env = RopeFlattenEnv(observation_mode='key_point',
                         action_mode='picker',
                         num_picker=num_picker,
                         render=True,
                         headless=False,
                         horizon=75,
                         action_repeat=8,
                         render_mode='cloth',
                         num_variations=200,
                         use_cached_states=True,
                         deterministic=False)
    # num_picker = 4
    # env = ClothFoldEnv(
    #     observation_mode='key_point',
    #     action_mode='picker',
    #     num_picker=num_picker,
    #     render=True,
    #     headless=False,
    #     horizon=100,
    #     action_repeat=8,
    #     render_mode='cloth',
    #     num_variations=200,
    #     use_cached_states=True,
    #     deterministic=False)

    env = PourWaterPosControlEnv(
        observation_mode='key_point',
        horizon=75,
        render=True,
        headless=False,
        action_mode='direct',
        deterministic=False,
        render_mode='fluid')

    # test_keypoint(env, num_picker)
    test_random(env, 5)
