from softgym.envs.pass_water import PassWater1DEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym
from softgym.utils.visualization import save_numpy_as_gif
import time
import torchvision, torch

script = 'heuristic'

env = PassWater1DEnv(
    observation_mode='cam_rgb',
    action_mode='direct',
    render=True,
    headless=True,
    horizon=75,
    action_repeat=8,
    render_mode='fluid',
    delta_reward=False,
    deterministic=False)

def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])

imgs = []
for _ in range(16):
    env.reset()
    particle_y = get_particle_max_y()
    img = []
    for i in range(40):
        print('step: ', i)
        
        if np.abs(env.height - particle_y) > 0.2: # small water
            print("small")
            action = np.array([0.02])
        elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
            print("medium")
            action = np.array([0.01])
        else:
            print("large")
            action = np.array([0.003])

        if np.abs(env.glass_x - env.terminal_x) < 0.1:
            print("achieving target!")
            action = np.array([0]) 

        if script == 'random':
            action = env.action_space.sample()
            action = np.clip(action, a_min = 0, a_max=np.inf)
     
     
        _, reward, _, info = env.step(action)
        print("glass x {} reward {}".format(env.glass_x, reward))
        img.append(env.get_image(128, 128))

    imgs.append(img)

all_frames = imgs
all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

from os import path as osp
save_name = 'pass_water_heuristic' + '.gif'
save_numpy_as_gif(np.array(grid_imgs), osp.join('./data/video/env_demos', save_name))