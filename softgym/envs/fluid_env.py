import numpy as np
import gym
import pyflex
from softgym.envs.flex_env import FlexEnv

class FluidEnv(FlexEnv):

    def __init__(self, deterministic = False, render_mode = 'particle', **kwargs):
        self.dim_shape_state = 14 # dimension of a shape object in Flex
        self.dim_position = 4
        self.dim_velocity = 3
        self.debug = False
        self.deterministic = deterministic
        assert render_mode in ['particle', 'fluid']
        self.render_mode = 0 if render_mode == 'particle' else 1
        super().__init__(**kwargs)


    def sample_fluid_params(self, fluid_param_dic = None):
        '''
        sample params for the fluid.
        '''
        params = {}
        params['radius_range'] = [0.09, 0.11] # 1.0
        params['rest_dis_coef_range'] = [0.4, 0.6] # 0.55
        params['cohension_range'] = [0.015, 0.025] # large, like mud. // 0.02f;
        params['viscosity_range'] = [1.5, 2.5] # //2.0f;
        params['surfaceTension_range'] = [0., 0.1] # 0.0
        params['adhesion_range'] = [0., 0.002] # how fluid adhead to shape. do not set to too large! # 0.0
        params['vorticityConfinement_range'] = [39.99, 40.01] # // 40.0f;
        params['solidPressure_range'] = [0., 0.01] #//0.f;

       
        params['radius'] = self.rand_float(params['radius_range'][0], params['radius_range'][1])
        params['rest_dis_coef'] = self.rand_float(params['rest_dis_coef_range'][0], params['rest_dis_coef_range'][1])
        params['cohesion'] = self.rand_float(params['cohension_range'][0], params['cohension_range'][1])
        params['viscosity'] = self.rand_float(params['viscosity_range'][0], params['viscosity_range'][1])
        params['surfaceTension'] = self.rand_float(params['surfaceTension_range'][0], params['surfaceTension_range'][1])
        params['adhesion'] = self.rand_float(params['adhesion_range'][0], params['adhesion_range'][1])
        params['vorticityConfinement'] = self.rand_float(params['vorticityConfinement_range'][0], params['vorticityConfinement_range'][1])
        params['solidpressure'] = self.rand_float(params['solidPressure_range'][0], params['solidPressure_range'][1])
      
        self.fluid_params = params

        # num of particles in x,y,z-axis
        self.fluid_params['dim_x_range'] = 4, 6
        self.fluid_params['dim_y_range'] = 16, 20
        self.fluid_params['dim_z_range'] = 4, 6 
     
        self.fluid_params['dim_x'] = self.rand_int(self.fluid_params['dim_x_range'][0], self.fluid_params['dim_x_range'][1]) 
        self.fluid_params['dim_y'] = self.rand_int(self.fluid_params['dim_y_range'][0], self.fluid_params['dim_y_range'][1])
        self.fluid_params['dim_z'] = self.rand_int(self.fluid_params['dim_z_range'][0], self.fluid_params['dim_z_range'][1])
        
        # center of the glass floor. lower corner of the water fluid grid along x,y,z-axis. 
        fluid_radis = params['radius'] * params['rest_dis_coef']
        if not self.deterministic:
            self.x_center = self.rand_float(-0.2, 0.2) 
        else:
            self.x_center = 0
        self.fluid_params['x'] = self.x_center - (self.fluid_params['dim_x']-1)/1.*fluid_radis 
        self.fluid_params['y'] = fluid_radis/1.5 + 0.05
        self.fluid_params['z'] = 0. - (self.fluid_params['dim_z'])/1.2*fluid_radis 

        # overwrite the parameters speicified by user
        if self.deterministic:
            for k in fluid_param_dic:
                self.fluid_params[k] = fluid_param_dic[k]
        
        return np.array([params['radius'], params['rest_dis_coef'], params['cohesion'], params['viscosity'], 
            params['surfaceTension'], params['adhesion'], params['vorticityConfinement'], params['solidpressure'], 
            self.fluid_params['x'], self.fluid_params['y'], self.fluid_params['z'], 
            self.fluid_params['dim_x'], self.fluid_params['dim_y'], self.fluid_params['dim_z']])

    def set_scene(self, fluid_param_dic = None):
        '''
        child envs can pass in specific fluid params through fluid param dic.
        '''
        # sample fluid properties.
        fluid_params = self.sample_fluid_params(fluid_param_dic)
        print(self.fluid_params)

        # set camera parameters. 
        self.initialize_camera()
        camera_x, camera_y, camera_z = self.camera_params['pos'][0], self.camera_params['pos'][1], self.camera_params['pos'][2]
        camera_ax, camera_ay, camera_az = self.camera_params['angle'][0], self.camera_params['angle'][1], self.camera_params['angle'][2]
        camera_params = np.array([camera_x, camera_y, camera_z, camera_ax, camera_ay, camera_az, 
            self.camera_width, self.camera_height, self.render_mode])
        
        # create fluid
        scene_params = np.concatenate((fluid_params, camera_params))
        # print("right before set_scene")
        pyflex.set_scene(11, scene_params, 0)
        print("right after pyflex set scene: num particles are: ", pyflex.get_n_particles())
        # print("right after set_scene")
        # print("hahahah")

    def rand_float(self, lo, hi):
        return np.random.rand() * (hi - lo) + lo

    def rand_int(self, lo, hi):
        return np.random.randint(lo, hi)

    def quatFromAxisAngle(self, axis, angle):
        '''
        given a rotation axis and angle, return a quatirian that represents such roatation.
        '''
        axis /= np.linalg.norm(axis)

        half = angle * 0.5
        w = np.cos(half)

        sin_theta_over_two = np.sin(half)
        axis *= sin_theta_over_two

        quat = np.array([axis[0], axis[1], axis[2], w])

        return quat

    def set_video_recording_params(self):
        """
        Set the following parameters if video recording is needed:
            video_idx_st, video_idx_en, video_height, video_width
        """
        self.video_height = 240
        self.video_width = 320

    


