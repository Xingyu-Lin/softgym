import numpy as np
import gym
import pyflex
from softgym.envs.flex_env import FlexEnv


class FluidTorusEnv(FlexEnv):

    def __init__(self, render_mode='particle', **kwargs):
        self.debug = False
        assert render_mode in ['particle', 'torus']
        self.render_mode = 0 if render_mode == 'particle' else 1
        super().__init__(**kwargs)

    def set_scene(self, config, states=None):
        '''
        child envs can pass in specific fluid params through fluid param dic.
        '''
        radius, rest_dist_coef, num, size = config['torus']['radius'], config['torus']['rest_dis_coef'], \
                config['torus']['num'], config['torus']['size']
        lower_x, torus_height, lower_z = config['torus']['lower_x'], config['torus']['height'], config['torus']['lower_z']
        torus_params = np.array([radius, rest_dist_coef, num, size, lower_x, torus_height, lower_z, 
            config['static_friction'], config['dynamic_friction']])

        # set camera parameters. 
        self.initialize_camera()
        camera_name = config.get('camera_name', self.camera_name)
        camera_params = np.array([*self.camera_params[camera_name]['pos'],
                                  *self.camera_params[camera_name]['angle'], self.camera_width, self.camera_height, self.render_mode])

        # create fluid
        scene_params = np.concatenate((torus_params, camera_params))

        env_idx = 4
        if self.version == 2:
            robot_params = [0]
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)
        
        self.particle_num = pyflex.get_n_particles()
        # print("particle num: ", self.particle_num)
        # print("particle pos: ", pyflex.get_positions().reshape([-1, 4])[:, :3])

    def rand_float(self, lo, hi):
        return np.random.rand() * (hi - lo) + lo

    def rand_int(self, lo, hi):
        return np.random.randint(lo, hi)

    def set_video_recording_params(self):
        """
        Set the following parameters if video recording is needed:
            video_idx_st, video_idx_en, video_height, video_width
        """
        self.video_height = 240
        self.video_width = 320

    def _get_info(self):
        return {}

    def _get_current_torus_height(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        return np.max(pos[:, 1])
