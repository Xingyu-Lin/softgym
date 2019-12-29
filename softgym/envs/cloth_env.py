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
        self.prev_rot = np.array([0.0, 0.0])
        self.init_rot = self.prev_rot
        self.prev_middle = None
        self.prev_dist = None

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

    def set_scene(self, initX=0.0, initY=-1.0, initZ=3.0, sizex=64.0, sizey=32.0, stretch=0.9, bend=1.0, shear=0.9,
                  render_mode=2,
                  cam_x=6.0, cam_y=8.0, cam_z=18.0, angle_x=0.0, angle_y=-np.deg2rad(20.0), angle_z=0.0,
                  width=960, height=720):
        camera_x, camera_y, camera_z = self.camera_params['pos'][0], \
                                       self.camera_params['pos'][1], \
                                       self.camera_params['pos'][2]
        camera_ax, camera_ay, camera_az = self.camera_params['angle'][0], \
                                          self.camera_params['angle'][1], \
                                          self.camera_params['angle'][2]
        print("cloth pos: {} {} {}".format(self.config['ClothPos']['x'], self.config['ClothPos']['y'],
                                           self.config['ClothPos']['z']))
                                           
        # params = np.array([self.config['ClothPos']['x'], self.config['ClothPos']['y'], self.config['ClothPos']['z'],
        #                    self.config['ClothSize']['x'], self.config['ClothSize']['y'],
        #                    self.config['ClothStiff']['stretch'], self.config['ClothStiff']['bend'],
        #                    self.config['ClothStiff']['shear'],
        #                    self.config['RenderMode'], self.config['Camera']['xPos'], self.config['Camera']['yPos'],
        #                    self.config['Camera']['zPos'],
        #                    self.config['Camera']['xAngle'], self.config['Camera']['yAngle'],
        #                    self.config['Camera']['zAngle'],
        #                    self.config['Camera']['width'], self.config['Camera']['height']])

        params = np.array([self.config['ClothPos']['x'], self.config['ClothPos']['y'], self.config['ClothPos']['z'],
                           self.config['ClothSize']['x'], self.config['ClothSize']['y'],
                           self.config['ClothStiff']['stretch'], self.config['ClothStiff']['bend'],
                           self.config['ClothStiff']['shear'], self.config['RenderMode'], camera_x, camera_y, 
                           camera_z, camera_ax, camera_ay, camera_az, self.camera_width, self.camera_height])

        self.params = params # YF NOTE: need to save the params for sampling goals
        
        # params = np.array([initX, initY, initZ, sizex, sizey, stretch, bend, shear, render_mode,
        #                  camera_x, camera_y, camera_z, camera_ax, camera_ay, camera_az, self.camera_width, self.camera_height])
        pyflex.set_scene(9, params, 0)

    def addSpheres(self, radii=0.1, initPos1=[1.5, 0.25, 3.4], initPos2=[1.5, 0.25, 3.6]):
        grip1sphere1pos = initPos1
        grip1sphere2pos = initPos1
        grip2sphere1pos = initPos2
        grip2sphere2pos = initPos2
        grip1sphere1pos[0] = grip1sphere1pos[0] + 0.25
        grip1sphere2pos[0] = grip1sphere2pos[0] - 0.25
        grip2sphere1pos[0] = grip2sphere1pos[0] + 0.25
        grip2sphere2pos[0] = grip2sphere2pos[0] - 0.25
        pyflex.add_sphere(radii, grip1sphere1pos, [1, 0, 0, 0])
        pyflex.add_sphere(radii, grip1sphere2pos, [1, 0, 0, 0])
        pyflex.add_sphere(radii, grip2sphere1pos, [1, 0, 0, 0])
        pyflex.add_sphere(radii, grip2sphere2pos, [1, 0, 0, 0])
        self.prev_middle = np.array([initPos1, initPos2])
        self.init_mid = self.prev_middle
        # print("init prev_mid: {}".format(self.prev_middle))
        self.prev_dist = np.array([0.0, 0.0])
        self.init_dist = self.prev_dist
        self.radii = radii

    """
    sphere manipulation function just in case
    """

    def sphereStep(self, action, last_pos):
        action = np.reshape(action, [-1, 5])
        cur_pos = np.array(pyflex.get_shape_states())
        cur_middle = np.zeros((2, 3))
        for i in range(0, 2):
            actionPos = action[i, 0:3]
            actionRot = action[i, 3]
            actionDist = action[i, 4]

            new_rot = actionRot + self.prev_rot[i]
            new_dist = actionDist + self.prev_dist[i]
            new_dist = min(new_dist, 0.6)
            new_dist = max(new_dist, 0.2)
            last_pos = np.reshape(last_pos, [-1, 14])
            cur_pos = np.reshape(cur_pos, [-1, 14])
            # cur_middle = cur_pos[0][0:3] +  np.array([self.prev_dist[i]/2 * np.cos(new_rot), 0, self.prev_dist[i]/2 * np.sin(new_rot)])
            # print("cur middle: {} prev_mid: {}".format(cur_middle, self.prev_middle))
            cur_middle[i, :] = self.prev_middle[i, :] + actionPos
            offset = np.array([new_dist / 2 * np.cos(new_rot), 0.0, new_dist / 2 * np.sin(new_rot)])
            cur_pos[i * 2][0:3] = cur_middle[i, :] - offset
            cur_pos[i * 2 + 1][0:3] = cur_middle[i, :] + offset
            cur_pos[i * 2][3:6] = last_pos[i * 2][0:3]
            cur_pos[i * 2 + 1][3:6] = last_pos[i * 2 + 1][0:3]

            self.prev_rot[i] = new_rot
            self.prev_dist[i] = new_dist
        #print("prev dists: {}".format(self.prev_dist))
        if self.checkSphereCollisions(cur_pos[0][0:3], cur_pos[1][0:3], cur_pos[2][0:3], cur_pos[3][0:3]):
            pyflex.set_shape_states(cur_pos.flatten())
            self.prev_middle = cur_middle
            self.prev_middle[0, 1] = max(self.prev_middle[0, 1], 0.1)
            self.prev_middle[1, 1] = max(self.prev_middle[1, 1], 0.1)
        else:
            self.prev_rot = self.prev_rot - action[:, 3]
            self.prev_dist = self.prev_dist - action[:, 4]

    def sphere_reset(self):
        self.prev_middle = self.init_mid
        self.prev_rot = self.init_rot
        self.prev_dist = self.init_dist
        # print("prev_mid reset: {}".format(self.prev_middle))


    def get_state(self):
        cur_state = super().get_state()
        cur_state['middle'] = self.prev_middle
        cur_state['rot'] = self.prev_rot
        cur_state['dist'] = self.prev_dist
        return cur_state

    def set_state(self, state_dict):
        self.prev_middle = state_dict['middle']
        self.prev_dist = state_dict['dist']
        self.prev_rot = state_dict['rot']
        super().set_state(state_dict)

    def checkSphereCollisions(self, sp1, sp2, sp3, sp4):
        sp13dist = np.linalg.norm(sp1 - sp3)
        sp14dist = np.linalg.norm(sp1 - sp4)
        sp23dist = np.linalg.norm((sp2 - sp3))
        sp24dist = np.linalg.norm(sp2 - sp4)

        return sp13dist > self.radii and sp14dist > self.radii and sp23dist > self.radii and sp24dist > self.radii
