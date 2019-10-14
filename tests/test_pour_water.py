import gym
import numpy as np
import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import os

env = PourWaterPosControlEnv(observation_mode = 'cam_img', action_mode = 'direct')

timestep = 500
move_part = int(0.3 * timestep)
stable_part = int(0.1 * timestep)

v = 0.07
y = 0
dt = 0.1
x = env.glass_floor_centerx
total_rotate = 0.6* np.pi
dim_position = 4
dim_velocity = 3
dim_shape_state = 14

n_particles = pyflex.get_n_particles()
n_shapes = pyflex.get_n_shapes()
n_rigids = pyflex.get_n_rigids()
n_rigidPositions = pyflex.get_n_rigidPositions()
positions = np.zeros((timestep, n_particles, dim_position))
velocities = np.zeros((timestep, n_particles, dim_velocity))
shape_states = np.zeros((timestep, n_shapes, dim_shape_state))

env.reset()
for i in range(timestep):
    if i < stable_part:
        action = np.array([x, y, 0])

    elif stable_part <= i < move_part + stable_part:
        y = y + v * dt
        action = np.array([x, y, 0.])

    else:
        theta = (i - move_part - stable_part) / float(timestep - move_part - stable_part) * total_rotate
        action = np.array([x, y, theta])

    positions[i] = pyflex.get_positions().reshape(-1, dim_position)
    velocities[i] = pyflex.get_velocities().reshape(-1, dim_velocity)
    shape_states[i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)   
    _, reward, _, _ = env.step(action)

    print("step {} reward {}".format(i, reward))

    

# env.reset()
# for i in range(timestep):
#     pyflex.set_positions(positions[i])
#     pyflex.set_shape_states(shape_states[i, :-1]) ### render removes front wall

#     pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

# pyflex.clean()