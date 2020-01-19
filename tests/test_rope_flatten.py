import gym
import numpy as np
import pyflex
from softgym.envs.rope_flatten import RopeFlattenEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt


def test_random(env):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    # test_picker()
    num_picker = 4
    env = RopeFlattenEnv(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=True,
        horizon=75,
        action_repeat=8,
        render_mode='cloth',
        num_variations=50,
        use_cached_states=False,
        deterministic=False)
    test_random(env)
