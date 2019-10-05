import os
import numpy as np
import pyflex
import time

import scipy.spatial as spatial
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


dt = 1. / 60.

time_step = 300
time_step_rest = 30
dim_position = 4
dim_velocity = 3
dim_shape_state = 14

border = 0.025
height = 1.3


def calc_box_init(dis_x, dis_z):
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])
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


def calc_shape_states(x_curr, x_last, box_dis):
    dis_x, dis_z = box_dis
    quat = np.array([1., 0., 0., 0.])

    states = np.zeros((5, dim_shape_state))

    states[0, :3] = np.array([x_curr, border/2., 0.])
    states[0, 3:6] = np.array([x_last, border/2., 0.])

    states[1, :3] = np.array([x_curr-(dis_x+border)/2., (height+border)/2., 0.])
    states[1, 3:6] = np.array([x_last-(dis_x+border)/2., (height+border)/2., 0.])

    states[2, :3] = np.array([x_curr+(dis_x+border)/2., (height+border)/2., 0.])
    states[2, 3:6] = np.array([x_last+(dis_x+border)/2., (height+border)/2., 0.])

    states[3, :3] = np.array([x_curr, (height+border)/2., -(dis_z+border)/2.])
    states[3, 3:6] = np.array([x_last, (height+border)/2., -(dis_z+border)/2.])

    states[4, :3] = np.array([x_curr, (height+border)/2., (dis_z+border)/2.])
    states[4, 3:6] = np.array([x_last, (height+border)/2., (dis_z+border)/2.])

    states[:, 6:10] = quat
    states[:, 10:] = quat

    return states



pyflex.init()


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

### set scene

# fluid params:
# px_f: [0.0, 0.3]
# py_f: [0.1, 0.2]
# pz_f: [0.0, 0.3]
# sx_f: [8, 12]
# sy_f: [15, 20]
# sz_f: 3

# rigid params:
# px_r: []
# py_r: []
# pz_r: []
# sx_r: []
# sy_r: []
# sz_r: []

sx_f = rand_int(10, 12)
sy_f = rand_int(10, 12)
sz_f = 3
x_center = rand_float(-0.2, 0.2)
px_f = x_center - (sx_f - 1) / 2. * 0.055
py_f = 0.055 / 2. + border + 0.01
pz_f = 0. - (sz_f - 1) / 2. * 0.055
box_dis_x = sx_f * 0.055 + rand_float(0., 0.3)
box_dis_z = 0.125

sx_r = rand_float(0.2, 0.399)
sy_r = rand_float(0.2, 0.399)
# sx_r = 0.2
# sy_r = 0.2
sz_r = 0.15
px_r = x_center - sx_r / 2.
py_r = py_f + sy_f * 0.052
pz_r = -sz_r / 2.

scene_params = np.array([
    px_f, py_f, pz_f, sx_f, sy_f, sz_f,
    px_r, py_r, pz_r, sx_r, sy_r, sz_r,
    box_dis_x, box_dis_z])

print("scene_params", scene_params)
pyflex.set_scene(8, scene_params, 0)

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
v_box = 0

for i in range(time_step):
    x_box_last = x_box
    x_box += v_box * dt
    v_box += rand_float(-0.15, 0.15) - x_box * 0.1
    shape_states_ = calc_shape_states(x_box, x_box_last, scene_params[-2:])
    pyflex.set_shape_states(shape_states_)

    positions[i] = pyflex.get_positions().reshape(-1, dim_position)
    velocities[i] = pyflex.get_velocities().reshape(-1, dim_velocity)
    shape_states[i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)

    if i == 0:
        print(np.min(positions[i], 0), np.max(positions[i], 0))
        print(x_box, box_dis_x, box_dis_z)

    pyflex.step()


pyflex.set_scene(8, scene_params, 0)
for i in range(len(boxes)-1):
    halfEdge = boxes[i][0]
    center = boxes[i][1]
    quat = boxes[i][2]
    pyflex.add_box(halfEdge, center, quat)


des_dir = 'test_FluidIceShake'
os.system('mkdir -p ' + des_dir)

for i in range(time_step):
    pyflex.set_positions(positions[i])
    pyflex.set_shape_states(shape_states[i, :-1])

    pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

pyflex.clean()
