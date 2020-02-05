import gym
import numpy as np
import pyflex
from softgym.envs.cloth_drop import ClothDropEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
import torch, torchvision, cv2


def test_picker(env, num_picker=2):
    imgs = []
    for _ in range(1):
        env.reset()

        fast_move_steps = 30
        for i in range(fast_move_steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 0] = 0.03
            env.step(action)
            imgs.append(env.render('rgb_array'))

        slow_move_steps = 30
        for i in range(slow_move_steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 0] = -0.02
            env.step(action)
            imgs.append(env.render('rgb_array'))

        let_go_steps = 100
        for i in range(let_go_steps):
            action = np.zeros((num_picker, 4))
            env.step(action)
            imgs.append(env.render('rgb_array'))

    num = 8
    show_imgs = []
    factor = len(imgs) // num
    for i in range(num):
        img = imgs[i * factor].transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))

    grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs=grid_imgs[:, :, ::-1]
    cv2.imwrite('cloth_drop.jpg', grid_imgs)

        # for i in range(50):
        #     print('step: ', i)
        #     action = np.zeros((num_picker, 4))
        #     if i < 10:
        #         action[:, 0] = 0.02
        #         action[:, 3] = 1.
        #     elif i < 15:
        #         action[:, 0] = -0.02
        #         action[:, 3] = 1.
        #     elif i < 40:
        #         action[:, 3] = 0.
        #     obs, reward, _, _ = env.step(action)
        #     print(reward)


def test_random(env, N=5):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            action = action.reshape(-1, 4)
            _, reward, _, _ = env.step(action.flatten())
            print(reward)


if __name__ == '__main__':
    num_picker = 2
    env = ClothDropEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=1000,
        action_repeat=1,
        render_mode='cloth',
        num_variations=1000,
        use_cached_states=True,
        deterministic=True)
    # test_random(env)
    test_picker(env)
