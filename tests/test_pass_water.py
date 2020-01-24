from softgym.envs.pass_water import PassWater1DEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym
from softgym.utils.visualization import save_numpy_as_gif
import time

num_picker = 2
script = ''

env = PassWater1DEnv(
    observation_mode='cam_rgb',
    action_mode='direct',
    render=True,
    headless=False,
    horizon=75,
    action_repeat=8,
    render_mode='fluid',
    delta_reward=False,
    deterministic=False)


imgs = []
for _ in range(5):
    env.reset()
    for i in range(50):
        print('step: ', i)
        action = np.array([0.01])
        if script == 'random':
            action = env.action_space.sample()
            action = np.clip(action, a_min = 0, a_max=np.inf)
        _, reward, _, info = env.step(action)
        print("glass x {} reward {}".format(env.glass_x, reward))
        img = env.render(mode='rgb_array')
        imgs.append(img)
        time.sleep(0.1)

# fp_out = './videos/flatten_picker_random_{}.gif'.format(num_picker)
# save_numpy_as_gif(np.array(imgs), fp_out)