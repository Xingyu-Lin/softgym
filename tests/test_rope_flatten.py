import gym
import numpy as np
import pyflex
from softgym.envs.rope_flatten import RopeFlattenEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
import torch, torchvision, cv2

def test_random(env):
    N = 1
    imgs = []
    for _ in range(N):
        env.reset()
        
        pos = pyflex.get_positions().reshape((-1, 4))
        corner1 = pos[0][:3]
        corner2 = pos[-1][:3]

        print("corner1: ", corner1)
        print("corner2: ", corner2)

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = corner1 - picker_pos[0]
        diff2 = corner2 - picker_pos[1]

        steps = 50
        for i in range(steps):
            action = np.zeros((2, 4))
            action[0, :3] = diff1 / steps
            action[1, :3] = diff2 / steps
            env.step(action)
            imgs.append(env.render('rgb_array'))


        print("=" * 50, "move to corner done!", "=" * 50)
        picker_pos, _ = env.action_tool._get_pos()
        print(picker_pos[0])
        print(picker_pos[1])

        target_pos_1 = np.array([2.5, 0.05, 0])
        target_pos_2 = np.array([-2.5, 0.05, 0])

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = target_pos_1 - picker_pos[0]
        diff2 = target_pos_2 - picker_pos[1]

        steps = 100
        for i in range(steps):
            action = np.ones((2, 4))
            action[0, :3] = diff1 / steps
            action[1, :3] = diff2 / steps
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
        cv2.imwrite('rope_flatten.jpg', grid_imgs)


    
        
        # for _ in range(10):
        #     action = env.action_space.sample()
        #     env.step(action)


if __name__ == '__main__':
    num_picker = 2
    env = RopeFlattenEnv(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=1,
        num_variations=1000,
        render_mode='cloth',
        use_cached_states=True,
        deterministic=True)

    test_random(env)
