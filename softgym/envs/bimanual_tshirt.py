import numpy as np
from gym import spaces
import pyflex
from softgym.envs.cloth_env import FlexEnv
from softgym.action_space.action_space import PickerPickPlace
from softgym.envs.bimanual_env import BimanualEnv
from copy import deepcopy
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R

class BimanualTshirtEnv(BimanualEnv):
    def __init__(self, use_depth=False, particle_radius=0.00625, picker_radius=0.01, **kwargs):
        super().__init__(use_depth=use_depth, particle_radius=particle_radius, picker_radius=picker_radius, **kwargs)
        self.action_distances = 1.8 * np.array([0.065, 0.13, 0.26])
        self.reset_act = np.array([0.,0.1,-0.5,0.,0.,0.1,-0.5,0.])
        self.reset_pos = np.array([0.,0.1,-0.5,0.,0.1,-0.5])

    def get_default_config(self):
        cam_pos, cam_angle = np.array([0.0, 0.65, 0.00]), np.array([0, -np.pi/2., 0.])
        config = {
            'pos': [0.01, 0.15, 0.01],
            'scale': 0.36,
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': 0.9,
            'mass': 0.05,
            'radius': self.cloth_particle_radius,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'drop_height': 0.0
        }

        return config


    def move_to_pos(self,new_pos):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos[:,:3] -= center[:3]
        pos[:,:3] += np.asarray(new_pos)
        pyflex.set_positions(pos)


    def rotate_particles(self, angle):
        r = R.from_euler('zyx', angle, degrees=True)
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()[:,:3]
        new_pos = r.apply(new_pos)
        new_pos = np.column_stack([new_pos,pos[:,3]])
        new_pos += center
        pyflex.set_positions(new_pos)


    def _set_to_flat(self):
        pyflex.set_positions(self.default_pos)
        self.rotate_particles([0,0,90])
        pyflex.step()

    
    def set_scene(self):
        config = self.config
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 5
        scene_params = np.concatenate([ config['pos'][:], [config['scale'], config['rot']], config['vel'][:], [config['stiff'], config['mass'], config['radius']],
                                camera_params['pos'][:], camera_params['angle'][:], [camera_params['width'], camera_params['height']] ])
        pyflex.set_scene(env_idx, scene_params, 0)
        self.default_pos = pyflex.get_positions().reshape(-1, 4)

    def reset(self,given_goal=None, given_goal_pos=None):
        self.set_scene()
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        self._set_to_flat()
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0, 0.1, 0])
            self.set_picker_pos(self.reset_pos)
        self.goal = given_goal 
        self.goal_pos = given_goal_pos 
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))

        self.render(mode='rgb_array')
        self._set_to_flat()
        self.move_to_pos([0,0.05,0])
        for i in range(10):
            pyflex.step()
            #self.render(mode='rgb_array')
        obs = self._get_obs()

        return obs