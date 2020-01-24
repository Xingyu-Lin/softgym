from softgym.envs.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
import numpy as np
from matplotlib import pyplot as plt
import softgym, gym
from softgym.utils.visualization import save_numpy_as_gif


num_picker = 2
script = 'random'

env = PourWaterPosControlGoalConditionedEnv(observation_mode = 'point_cloud', horizon = 75, 
        action_mode = 'direct', deterministic=True, render_mode = 'fluid', render = True, headless= False)

for i in range(5):
    print("right before reset")
    env.reset()
    env.set_to_goal(env.get_goal())
    img = env.get_image(960, 720)
    plt.imshow(img)
    # plt.show()
    plt.savefig('./imgs/pour_water_multitask_{}.png'.format(i))

env.reset()

timestep = env.horizon
move_part = 15
stable_part = int(0.0 * timestep)

v = 0.13
y = 0
dt = 0.1
x = env.glass_floor_centerx
total_rotate = 0.28* np.pi

# test a heuristic policy
print("right before reset")
env.reset()
print("total timestep: ", timestep)
for i in range(timestep):
    if i < stable_part:
        action = np.array([0, 0, 0])

    elif stable_part <= i < move_part + stable_part:
        y = v * dt
        action = np.array([0, y, 0.])

    elif i > move_part + stable_part and i < timestep - 30:
        theta = 1 / float(timestep - move_part - stable_part) * total_rotate
        action = np.array([0, 0, theta])

    else:
        action = np.array([0, 0 ,0])

    obs, reward, done, _ = env.step(action)

    # if i  == 250:
    #     # from matplotlib import pyplot as plt
    #     import cv2
    #     img = env.get_image(48, 48)
    #     cv2.imshow('test_img', img)
    #     cv2.waitKey(0)

    print("step {} reward {}".format(i, reward))
    if done:
        # env.end_record()
        
        print("done!")
        break

# fp_out = './videos/flatten_picker_random_{}.gif'.format(num_picker)
# save_numpy_as_gif(np.array(imgs), fp_out)