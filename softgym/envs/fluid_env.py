import numpy as np
import gym
import pyflex
from softgym.envs.flex_env import FlexEnv


class FluidEnv(FlexEnv):

    def __init__(self, render_mode='particle', **kwargs):
        self.debug = False
        assert render_mode in ['particle', 'fluid']
        self.render_mode = 0 if render_mode == 'particle' else 1
        super().__init__(**kwargs)

    def sample_fluid_params(self, fluid_param_dic):
        '''
        sample params for the fluid.
        '''
        params = fluid_param_dic
        self.fluid_params = fluid_param_dic

        # center of the glass floor. lower corner of the water fluid grid along x,y,z-axis. 
        fluid_radis = params['radius'] * params['rest_dis_coef']
        self.x_center = 0

        self.fluid_params['x'] = self.x_center - (self.fluid_params['dim_x'] - 3) / 1. * fluid_radis + 0.1
        self.fluid_params['y'] = fluid_radis / 2 + 0.05
        self.fluid_params['z'] = 0. - (self.fluid_params['dim_z'] - 2) * fluid_radis / 1.5

        return np.array([params['radius'], params['rest_dis_coef'], params['cohesion'], params['viscosity'],
                         params['surfaceTension'], params['adhesion'], params['vorticityConfinement'], params['solidpressure'],
                         self.fluid_params['x'], self.fluid_params['y'], self.fluid_params['z'],
                         self.fluid_params['dim_x'], self.fluid_params['dim_y'], self.fluid_params['dim_z']])

    def set_scene(self, config, states=None):
        '''
        child envs can pass in specific fluid params through fluid param dic.
        '''
        # sample fluid properties.
        fluid_params = self.sample_fluid_params(config['fluid'])

        # set camera parameters. 
        self.initialize_camera()
        camera_name = config.get('camera_name', self.camera_name)
        camera_params = np.array([*self.camera_params[camera_name]['pos'],
                                  *self.camera_params[camera_name]['angle'], self.camera_width, self.camera_height, self.render_mode])

        # create fluid
        scene_params = np.concatenate((fluid_params, camera_params))

        env_idx = 1
        if self.version == 2:
            robot_params = []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)
        
        self.particle_num = pyflex.get_n_particles()

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

    def _get_current_water_height(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        return np.max(pos[:, 1])
