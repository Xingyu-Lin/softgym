import os
import numpy as np
import pyflex
import time


time_step = 120
des_dir = 'test_FluidFall'
os.system('mkdir -p ' + des_dir)

pyflex.init()

scene_params = np.array([])
pyflex.set_scene(4, scene_params, 0)

for i in range(time_step):
    pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

pyflex.clean()
