from softgym.envs.cloth_flatten_multitask import ClothFlattenGoalConditionedEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym

softgym.register_flex_envs()
# env = ClothFlattenPointControlGoalConditionedEnv('key_point', 'sphere', render_mode='particle')
env = gym.make('ClothFlattenSphereControlGoalConditioned-v0')

for i in range(5):
    env.reset()
    env.set_to_goal(env.get_goal())
    img = env.get_image(960, 720)
    plt.imshow(img)
    plt.show()

env.reset(dropPoint=100)
print("reset, entering loop")
haveGrasped = False


for i in range(0, 700):
    if env.prev_middle[0, 1] > 0.11 and not haveGrasped:
        obs, reward, _, _ = env.step(np.array([0., -0.001, 0, 0, 0.01] * 2))
        print("reward: {}".format(reward))
    elif not haveGrasped:
        obs, reward, _, _ = env.step(np.array([0., -0.001, 0, 0, -0.01] * 2))
        print("reward: {}".format(reward))
        if env.prev_dist[0] < 0.21:
            haveGrasped = True
    else:
        obs, reward, _, _ = env.step(np.array([0., 0.001, 0, 0, 0] * 2))
        print("reward: {}".format(reward))