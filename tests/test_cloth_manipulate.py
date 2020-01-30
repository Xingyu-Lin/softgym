from softgym.envs.cloth_manipulate import ClothManipulateEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym
from softgym.utils.visualization import save_numpy_as_gif


num_picker = 2
script = 'random'

env = ClothManipulateEnv(
    observation_mode='point_cloud',
    action_mode='picker',
    num_picker=num_picker,
    render=True,
    headless=False,
    horizon=75,
    action_repeat=8,
    render_mode='cloth',
    num_variations=200,
    deterministic=False)

for i in range(5):
    print("right before reset")
    env.reset()
    env.set_to_goal(env.get_goal())
    img = env.get_image(960, 720)
    plt.imshow(img)
    plt.show()
    # plt.savefig('./imgs/cloth_manipulation_goal_{}.png'.format(i))

imgs = []
for _ in range(5):
    env.reset()
    for i in range(50):
        print('step: ', i)
        action = np.zeros((num_picker, 4))
        if i < 12:
            action[:, 1] = -0.01
            action[:, 3] = 0
        elif i < 30:
            action[:, 1] = 0.01
            action[:, 3] = 1
        elif i < 40:
            action[:, 3] = 0
        if script == 'random':
            action = env.action_space.sample()
        env.step(action)
        img = env.render(mode='rgb_array')
        imgs.append(img)

# fp_out = './videos/flatten_picker_random_{}.gif'.format(num_picker)
# save_numpy_as_gif(np.array(imgs), fp_out)