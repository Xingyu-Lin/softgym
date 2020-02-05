from softgym.envs.pass_water import PassWater1DEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym
from softgym.utils.visualization import save_numpy_as_gif
import time
import torchvision, torch
import cv2

script = 'heuristic'

env = PassWater1DEnv(
    observation_mode='cam_rgb',
    action_mode='direct',
    render=True,
    headless=False,
    horizon=100,
    action_repeat=8,
    num_variations=1000,
    render_mode='fluid',
    delta_reward=False,
    deterministic=True)

def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])


for _ in range(1):
    env.reset()
    particle_y = get_particle_max_y()
    imgs = []
    for i in range(31):
        print('step: ', i)
        
        if np.abs(env.height - particle_y) > 0.2: # small water
            print("small")
            action = np.array([0.02])
        elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
            print("medium")
            action = np.array([0.017])
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
        imgs.append(env.render(mode='rgb_array'))


num = 8
show_imgs = []
factor = len(imgs) // num
for i in range(num):
    img = imgs[i * factor].transpose(2, 0, 1)
    print(img.shape)
    show_imgs.append(torch.from_numpy(img.copy()))

grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
grid_imgs=grid_imgs[:, :, ::-1]
cv2.imwrite('pass_water.jpg', grid_imgs)

# all_frames = imgs
# all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
# grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

# from os import path as osp
# save_name = 'pass_water_heuristic' + '.gif'
# save_numpy_as_gif(np.array(grid_imgs), osp.join('./data/video/env_demos', save_name))