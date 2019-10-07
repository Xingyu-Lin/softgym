import gym
import numpy as np
import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import os

env = PourWaterPosControlEnv(observation_mode = 'key_point', action_mode = 'direct')

des_dir = 'test_FluidShake'
os.system('mkdir -p ' + des_dir)    

timestep = 500
move_part = int(0.3 * timestep)

v = 0.05
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
    if i < move_part:
        y = y + v * dt
        action = np.array([x, y, 0.])
        env.step(action)

    else:
        theta = (i - move_part) / float(timestep - move_part) * total_rotate
        action = np.array([x, y, theta])
        env.step(action)

    positions[i] = pyflex.get_positions().reshape(-1, dim_position)
    velocities[i] = pyflex.get_velocities().reshape(-1, dim_velocity)
    shape_states[i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)
    

env.reset()
for i in range(timestep):
    pyflex.set_positions(positions[i])
    pyflex.set_shape_states(shape_states[i, :-1]) ### render removes front wall

    pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

pyflex.clean()