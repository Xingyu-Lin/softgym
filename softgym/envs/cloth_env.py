import numpy as np
import gym
import pyflex
from scipy.spatial.transform import Rotation as R
from softgym.envs.flex_env import FlexEnv
import yaml

class ClothEnv(FlexEnv):

    def __init__(self, configFile, randomized=True, horizon=100, device_id=-1):
        self.config = yaml.load(configFile)
        print(yaml.dump(self.config))
        self.initialize_camera()
        self.camera_width = 960
        self.camera_height = 720
        super().__init__()
        self.is_randomized = randomized
        self.horizon = 100
        self.prev_rot = 0
        self.prev_dist = 0

    def initialize_camera(self):
        '''
        set the camera width, height, ition and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        self.camera_params = {
            'pos': np.array([1.7, 2., 1]),
            'angle': np.array([0., -30 / 180. * np.pi, 0.]),
            'width': self.camera_width,
            'height': self.camera_height
        }
    """
    There's always the same parameters that you can set 
    """
    def set_scene(self, initX=0.0, initY=-1.0, initZ=3.0, sizex=64.0, sizey=32.0, stretch=0.9, bend=1.0, shear=0.9, render_mode=2,
                        cam__x = 6.0, cam_y = 8.0, cam_z = 18.0, angle_x = 0.0, angle_y = -np.deg2rad(20.0), angle_z = 0.0,
                         width = 960, height = 720):
        camera_x, camera_y, camera_z = self.camera_params['pos'][0], \
                                       self.camera_params['pos'][1], \
                                       self.camera_params['pos'][2]
        camera_ax, camera_ay, camera_az = self.camera_params['angle'][0], \
                                          self.camera_params['angle'][1], \
                                          self.camera_params['angle'][2]
        print("cloth pos: {} {} {}".format(self.config['ClothPos']['x'], self.config['ClothPos']['y'], self.config['ClothPos']['z']))
        params = np.array([self.config['ClothPos']['x'], self.config['ClothPos']['y'], self.config['ClothPos']['z'],
                           self.config['ClothSize']['x'], self.config['ClothSize']['y'],
                           self.config['ClothStiff']['stretch'], self.config['ClothStiff']['bend'], self.config['ClothStiff']['shear'],
                           self.config['RenderMode'], self.config['Camera']['xPos'], self.config['Camera']['yPos'], self.config['Camera']['zPos'],
                           self.config['Camera']['xAngle'], self.config['Camera']['yAngle'], self.config['Camera']['zAngle'],
                           self.config['Camera']['width'], self.config['Camera']['height']])
        #params = np.array([initX, initY, initZ, sizex, sizey, stretch, bend, shear, render_mode,
        #                  camera_x, camera_y, camera_z, camera_ax, camera_ay, camera_az, self.camera_width, self.camera_height])
        pyflex.set_scene(9, params, 0)


    def addSpheres(self, radii = 0.1, initPos1 = [1.5, 0.25, 3.5], initPos2 = [2.0, 0.25, 3.5]):
        pyflex.add_sphere(radii, initPos1, [1, 0, 0, 0])
        pyflex.add_sphere(radii, initPos2, [1, 0, 0, 0])

    """
    sphere manipulation function just in case
    """
    def sphereStep(self, action, last_pos):
        actionPos = action[:3]
        actionRot = action[3]
        actionDist = action[4]
        new_rot = actionRot+self.prev_rot
        new_dist = actionDist+self.prev_dist
        new_dist = min(new_dist, 0.4)
        new_dist = max(new_dist, 0.1)
        last_pos = np.reshape(last_pos, [-1, 14])
        cur_pos = np.array(pyflex.get_shape_states())
        cur_pos = np.reshape(cur_pos, [-1, 14])
        cur_middle = cur_pos[0][0:3] +  np.array([self.prev_dist/2 * np.cos(new_rot), 0, self.prev_dist/2 * np.sin(new_rot)])
        cur_pos[0][0:3] = cur_middle + actionPos -  np.array([new_dist/2 * np.cos(new_rot), 0, new_dist/2 * np.sin(new_rot)])
        cur_pos[1][0:3] = cur_middle + actionPos + np.array([new_dist/2 * np.cos(new_rot), 0, new_dist/2 * np.sin(new_rot)])
        cur_pos[0][3:6] = last_pos[0][0:3]
        cur_pos[1][3:6] = last_pos[1][0:3]
        cur_pos[0][1] = max(cur_pos[0][1], 0.06)
        cur_pos[1][1] = max(cur_pos[1][1], 0.06)

        pyflex.set_shape_states(cur_pos.flatten())
        self.prev_rot = new_rot
        self.prev_dist = new_dist
