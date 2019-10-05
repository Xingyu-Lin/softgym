import os
import numpy as np
import pyflex
import time

time_step = 40000
des_dir = 'test_Cloth'
os.system('mkdir -p ' + des_dir)

pyflex.init()
scene_params = np.array([])
pyflex.set_scene(9, scene_params, 0)

idx_c1 = 0
idx_c2 = 64 * 31

for i in range(time_step):
    pos = pyflex.get_positions()
    pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pos_after_step = pyflex.get_positions()
    pos_after_step = np.reshape(pos_after_step, [-1, 4])
    pos = np.reshape(pos, [-1, 4])
    original_pos = pos.copy()
    if i < 150:
        pos_after_step[idx_c1, :] = pos[idx_c1, :] + [0.02, 0., 0., 0, ]
        pos_after_step[idx_c2, :] = pos[idx_c2, :] + [0.02, 0., 0., 0.]
    else:
        pos_after_step[idx_c1, 3] = 1.
        pos_after_step[idx_c2, 3] = 1.

    if i < 150:
        pyflex.set_positions(pos_after_step.flatten())
    else:
        pyflex.set_positions(pos_after_step.flatten())

    # pyflex.set_positions(original_pos.flatten())
pyflex.clean()
