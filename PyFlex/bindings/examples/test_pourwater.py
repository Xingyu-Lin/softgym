import os
import numpy as np
import pyflex
import time
import torch

import scipy.spatial as spatial
# from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
The python code is in charge of creating the box, 
the Cpp code is in charge of creating the water.
'''


dt = 1. / 60.
des_dir = 'test_FluidShake'
os.system('mkdir -p ' + des_dir)

time_step = 500
time_step_rest = 30
dim_position = 4
dim_velocity = 3
dim_shape_state = 14  ### important note: this 14 is a magic number, it is even fixed in pyflex.cpp in pyflex_set_shape_states and pyflex_get_shape_states.
### this is really not good. Or perhaps Flex just need 14 parameters for all shape states?

border = 0.025 ### the thickness of the wall of the box
height = 0.5 ### this specify the y-axis

quatglobal = np.array([1., 0, 0., 0.])

'''
A simple box with interior [-halfHeight, +halfHeight] along each dimension 
center: g_buffers->shapePositions.push_back(Vec4(center.x, center.y, center.z, 0.0f)); line 445 at helper.h
quat: g_buffers->shapeRotations.push_back(quat); line 446 at helper.h
        
        y-axis (height)
          |
          |
          |
          |
          /----------- x-axis 
         /
        /
       /
     z-axis 
'''

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def calc_box_init(dis_x, dis_z):
    '''
    each wall of the real box is represented by a box object in FLex with really small thickness (determined by the param border)
    dis_x: the length of the x-axis edge of the box
    dis_z: the length of the z-axis edge of the box

    the halfEdge determines the center point of each wall.
    Note: this is merely setting the length of each dimension of the wall, but not the actual position of them.
    That's why leaf and right walls have exactly the same params, and so do front and back walls.   
    '''
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])
    # quat = quatglobal
    boxes = []

    # floor
    halfEdge = np.array([dis_x/2., border/2., dis_z/2.])
    boxes.append([halfEdge, center, quat])

    # left wall
    halfEdge = np.array([border/2., (height+border)/2., dis_z/2.])
    boxes.append([halfEdge, center, quat])

    # right wall
    boxes.append([halfEdge, center, quat])

    # back wall
    halfEdge = np.array([(dis_x+border*2)/2., (height+border)/2., border/2.])
    boxes.append([halfEdge, center, quat])

    # front wall
    boxes.append([halfEdge, center, quat])

    return boxes


def calc_shape_states(y_curr, y_last, box_dis):
    '''
    given the current and previous y center of the box,
    update the states of the 5 shapes that form the box: floor, left/right wall, back/front wall. 
    
    state:
    0-3: current (x, y, z) coordinate of the center point
    3-6: previous (x, y, z) coordinate of the center point
    6-10: current quat (rotation I guess?)
    10-14: previous quat (rotation I guess?)
    '''
    dis_x, dis_z = box_dis
    # quat = np.array([1., 0., 0., 0.])
    quat = quatFromAxisAngle([0, 0, -1.], 0.) 
    # quat = quatglobal

    # states of 5 walls
    states = np.zeros((5, dim_shape_state))

    # floor 
    states[0, :3] = np.array([x_center, y_curr, 0.])
    states[0, 3:6] = np.array([x_center, y_last, 0.])

    # left wall
    states[1, :3] = np.array([x_center-(dis_x+border)/2., (height+border)/2. + y_curr, 0.])
    states[1, 3:6] = np.array([x_center-(dis_x+border)/2., (height+border)/2. + y_last, 0.])

    # right wall
    states[2, :3] = np.array([x_center+(dis_x+border)/2., (height+border)/2. + y_curr, 0.])
    states[2, 3:6] = np.array([x_center+(dis_x+border)/2., (height+border)/2. + y_last, 0.])

    # back wall
    states[3, :3] = np.array([x_center, (height+border)/2. + y_curr, -(dis_z+border)/2.])
    states[3, 3:6] = np.array([x_center, (height+border)/2. + y_last, -(dis_z+border)/2.])

    # front wall
    states[4, :3] = np.array([x_center, (height+border)/2. + y_curr, (dis_z+border)/2.])
    states[4, 3:6] = np.array([x_center, (height+border)/2. + y_last, (dis_z+border)/2.])

    states[:, 6:10] = quat
    states[:, 10:] = quat

    return states

def rotate_box_states(y, prev_states, theta, box_dis):
    '''
    rotate the glass with angle theta.
    update the states of the 5 shapes that form the box: floor, left/right wall, back/front wall. 
    rotate the glass, where the center point is the center of the floor (bottom wall).
    
    state:
    0-3: current (x, y, z) coordinate of the center point
    3-6: previous (x, y, z) coordinate of the center point
    6-10: current quat (rotation I guess?)
    10-14: previous quat (rotation I guess?)
    '''
    dis_x, dis_z = box_dis
    quat_curr = quatFromAxisAngle([0, 0, -1.], theta) 

    w = dis_x / 2.
    h = height / 2.
    b = border / 2.

    # states of 5 walls
    states = np.zeros((5, dim_shape_state))

    for i in range(5):
        states[i][3:6] = prev_states[i][:3]
        states[i][10:] = prev_states[i][6:10]

    # floor: center position does not change
    states[0, :3] = np.array([x_center, y, 0.])

    # left wall: center must move right and move down. 
    states[1, :3] = np.array([x_center-(w+b)*np.cos(theta) + (h+b)*np.sin(theta),
         y + (w+b)*np.sin(theta) + (h+b)*np.cos(theta), 0.])

    # right wall
    states[2, :3] = np.array([x_center+ (w + b)*np.cos(theta) + (h+b)*np.sin(theta) ,
         y - (w+b)*np.sin(theta) + (h+b)*np.cos(theta), 0.])

    # back wall
    states[3, :3] = np.array([x_center + (h + b)*np.sin(theta), y + (h+b)*np.cos(theta), -(dis_z+border)/2.])

    # front wall
    states[4, :3] = np.array([x_center + (h + b)*np.sin(theta), y + (h+b)*np.cos(theta), (dis_z+border)/2.])

    states[:, 6:10] = quat_curr

    return states



pyflex.init()

use_gpu = torch.cuda.is_available()

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

### set scene
# dim_x: [8, 12]
# dim_y: [8, 12]
# dim_z: [4, 6]
# x: [0.0, 0.3]
# y: [0.1, 0.2]
# z: [0.0, 0.3]
'''
x, y, z, dim_x, dim_y and dim_z is used for creating the fluid grid (water).
(x, y, z) forms the origin of the grid.
dim_x is the number of grids on the x-axis.
we also need to specify the grid size, which is the magic number 0.055 here.
reason for 0.055: in the yz_fludishake.h, it has hard encoded the grid size at line 37 and line 41.
'''
# dim_x = rand_int(10, 12)
# dim_y = rand_int(15, 20)
dim_x = 5
dim_y = 20
dim_z = 3
x_center = rand_float(-0.2, 0.2)
x = x_center - (dim_x-1)/2.*0.055 ### here 0.055 is the grid step size.
y = 0.055/2. + border + 0.01
z = 0. - (dim_z-1)/2.*0.055
box_dis_x = dim_x * 0.055 + rand_float(0., 0.3)
box_dis_z = 0.2


scene_params = np.array([x, y, z, dim_x, dim_y, dim_z, box_dis_x, box_dis_z])
print("scene_params", scene_params)
pyflex.set_scene(6, scene_params, 0)

boxes = calc_box_init(box_dis_x, box_dis_z)

for i in range(len(boxes)):
    halfEdge = boxes[i][0]
    center = boxes[i][1]
    quat = boxes[i][2]
    pyflex.add_box(halfEdge, center, quat)


### read scene info
print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())
print("Num particles:", pyflex.get_phases().reshape(-1, 1).shape[0])
print("Phases:", np.unique(pyflex.get_phases()))

n_particles = pyflex.get_n_particles()
n_shapes = pyflex.get_n_shapes()
n_rigids = pyflex.get_n_rigids()
n_rigidPositions = pyflex.get_n_rigidPositions()

print("n_particles", n_particles)
print("n_shapes", n_shapes)
print("n_rigids", n_rigids)
print("n_rigidPositions", n_rigidPositions)

positions = np.zeros((time_step, n_particles, dim_position))
velocities = np.zeros((time_step, n_particles, dim_velocity))
shape_states = np.zeros((time_step, n_shapes, dim_shape_state))

x_box = x_center
v_box = 0 # note that v_box is just the velocity on the x-axis

y_box = 0
vy_box = 0.3

# simulation

move_part = 0.3 * time_step
total_rotate = 0.55 * np.pi

for i in range(time_step):
    if i < move_part:
        y_box_last = y_box
        y_box += vy_box * dt
        shape_states_ = calc_shape_states(y_box, y_box_last, scene_params[-2:])
        prev_states = shape_states_
    else:
        theta = (i - move_part) / float(time_step - move_part) * total_rotate
        shape_states_ = rotate_box_states(y_box, prev_states, theta, scene_params[-2:])
        prev_states = shape_states_

    pyflex.set_shape_states(shape_states_)

    positions[i] = pyflex.get_positions().reshape(-1, dim_position)
    velocities[i] = pyflex.get_velocities().reshape(-1, dim_velocity)
    shape_states[i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)

    if i == 0:
        print(np.min(positions[i], 0), np.max(positions[i], 0))
        print(x_box, box_dis_x, box_dis_z)

    pyflex.step()


# playback

pyflex.set_scene(6, scene_params, 0)
for i in range(len(boxes)-1):
    halfEdge = boxes[i][0]
    center = boxes[i][1]
    quat = boxes[i][2]
    pyflex.add_box(halfEdge, center, quat)

for i in range(time_step):
    pyflex.set_positions(positions[i])
    pyflex.set_shape_states(shape_states[i, :-1]) ### render removes front wall

    pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

pyflex.clean()
