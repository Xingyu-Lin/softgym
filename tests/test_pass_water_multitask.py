from softgym.multitask_envs_arxived.pass_water_multitask import PassWater1DGoalConditionedEnv
import numpy as np
from matplotlib import pyplot as plt

num_picker = 2
script = 'random'

env = PassWater1DGoalConditionedEnv(observation_mode = 'point_cloud', horizon = 75, 
        action_mode = 'direct', deterministic=False, render_mode = 'fluid', render = True, headless= False,
        num_variations=2)

for i in range(10):
    print("=" * 50)
    print("reset: ", i)
    env.reset()
    plt.imshow(env.get_image(960, 720))
    plt.show()
    env.set_to_goal(env.get_goal())
    img = env.get_image(960, 720)
    plt.imshow(img)
    plt.show()
    print("="*50)
    # plt.savefig('./imgs/pour_water_multitask_{}.png'.format(i))

def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])

script = ''
env.reset()
particle_y = get_particle_max_y()

for i in range(50):
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
     
    obs, reward, done, _ = env.step(action)


    print("step {} reward {}".format(i, reward))
    if done:
        print("done!")
        break