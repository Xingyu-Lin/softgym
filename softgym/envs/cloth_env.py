import numpy as np
import gym
import pyflex
from scipy.spatial.transform import Rotation as R
from softgym.envs.flex_env import FlexEnv
import yaml


class ClothEnv(FlexEnv):

    def __init__(self, config_file, randomized=True, device_id=-1, render_mode='particle', **kwargs):
        self.config = FlexEnv._load_config(config_file)
        self.render_mode = render_mode
        super().__init__(**kwargs)

        self.force = [{'idx': None, 'pos': None, 'strength': None, 'radius': None}, {'idx': None, 'pos': None, 'strength': None, 'radius': None}]
        self.init_force = [{'idx': None, 'pos': None, 'strength': None, 'radius': None}, {'idx': None, 'pos': None, 'strength': None, 'radius': None}]

        self.is_randomized = randomized
        self.prev_rot = np.array([0.0, 0.0])
        self.init_rot = self.prev_rot
        self.sticky_dist = 0.2
        self.prev_middle = None
        self.prev_dist = 1.0

        self.init_sticky = None
        self.sticky_idx = None

        self.xdim, self.ydim = self.config['ClothSize']['x'], self.config['ClothSize']['y']

    """
    There's always the same parameters that you can set 
    """

    def set_scene(self):
        self.initialize_camera()
        camera_x, camera_y, camera_z = self.camera_params['pos'][0], \
                                       self.camera_params['pos'][1], \
                                       self.camera_params['pos'][2]
        camera_ax, camera_ay, camera_az = self.camera_params['angle'][0], \
                                          self.camera_params['angle'][1], \
                                          self.camera_params['angle'][2]
        # print("cloth pos: {} {} {}".format(self.config['ClothPos']['x'], self.config['ClothPos']['y'],
        #                                    self.config['ClothPos']['z']))

        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        params = np.array([self.config['ClothPos']['x'], self.config['ClothPos']['y'], self.config['ClothPos']['z'],
                           self.config['ClothSize']['x'], self.config['ClothSize']['y'],
                           self.config['ClothStiff']['stretch'], self.config['ClothStiff']['bend'],
                           self.config['ClothStiff']['shear'], render_mode, camera_x, camera_y,
                           camera_z, camera_ax, camera_ay, camera_az, self.camera_width, self.camera_height])

        self.params = params  # YF NOTE: need to save the params for sampling goals

        pyflex.set_scene(9, params, 0)

    def add_spheres(self, sphere_radius=0.1, pick_point=0):
        # initial position of the spheres are near the pick point of the cloth
        random_particle_position = pyflex.get_positions().reshape((-1, 4))[pick_point]
        init_x = random_particle_position[0]
        init_y = 0.5
        init_z = random_particle_position[2]

        init_x += np.random.uniform(0, 0.1)
        init_z += np.random.uniform(0, 0.1)

        init_pos1 = [init_x, init_y, init_z]
        init_pos2 = [init_x, init_y, init_z + 0.2]

        grip1sphere1pos = init_pos1
        grip1sphere2pos = init_pos1
        grip2sphere1pos = init_pos2
        grip2sphere2pos = init_pos2
        grip1sphere1pos[0] = grip1sphere1pos[0] + 0.25
        grip1sphere2pos[0] = grip1sphere2pos[0] - 0.25
        grip2sphere1pos[0] = grip2sphere1pos[0] + 0.25
        grip2sphere2pos[0] = grip2sphere2pos[0] - 0.25
        pyflex.add_sphere(sphere_radius, grip1sphere1pos, [1, 0, 0, 0])
        pyflex.add_sphere(sphere_radius, grip1sphere2pos, [1, 0, 0, 0])
        pyflex.add_sphere(sphere_radius, grip2sphere1pos, [1, 0, 0, 0])
        pyflex.add_sphere(sphere_radius, grip2sphere2pos, [1, 0, 0, 0])
        self.prev_middle = np.array([init_pos1, init_pos2])
        self.init_mid = self.prev_middle
        # print("init prev_mid: {}".format(self.prev_middle))
        self.prev_dist = np.array([0.0, 0.0])
        self.init_dist = self.prev_dist
        self.radii = sphere_radius

    def addBlocks(self, radii=0.1, initPos1=[0.75, 0.25, -0.4], initPos2=[-0.75, 0.25, -0.2]):
        grip1sphere1pos = initPos1
        grip1sphere2pos = initPos1
        grip2sphere1pos = initPos2
        grip2sphere2pos = initPos2
        grip1sphere1pos[0] = grip1sphere1pos[0] + 0.25
        grip1sphere2pos[0] = grip1sphere2pos[0] - 0.25
        grip2sphere1pos[0] = grip2sphere1pos[0] + 0.25
        grip2sphere2pos[0] = grip2sphere2pos[0] - 0.25
        vol = np.array([radii, radii, radii]) / 2
        pyflex.add_box(vol, grip1sphere1pos, [1, 0, 0, 0])
        pyflex.add_box(vol, grip1sphere2pos, [1, 0, 0, 0])
        pyflex.add_box(vol, grip2sphere1pos, [1, 0, 0, 0])
        pyflex.add_box(vol, grip2sphere2pos, [1, 0, 0, 0])

        """
        pyflex.add_sphere(radii, grip1sphere1pos, [1, 0, 0, 0])
        pyflex.add_sphere(radii, grip1sphere2pos, [1, 0, 0, 0])
        pyflex.add_sphere(radii, grip2sphere1pos, [1, 0, 0, 0])
        pyflex.add_sphere(radii, grip2sphere2pos, [1, 0, 0, 0])
        """
        self.prev_middle = np.array([initPos1, initPos2])
        self.init_mid = self.prev_middle
        print("init prev_mid: {}".format(self.prev_middle))
        self.prev_dist = np.array([0.5, 0.5])
        self.init_dist = self.prev_dist
        self.radii = radii

    def addForce(self, radius=0.1, initPos1=[-0.1, 0.5, -0.6], initPos2=[-0, 1, 0.1, -0.2], strength=10000.0):
        self.force[0]['idx'] = pyflex.add_forcefield()
        self.force[0]['pos'] = np.array(initPos1)
        self.force[0]['radius'] = radius
        self.force[0]['strength'] = strength
        self.init_force[0]['idx'] = self.force[0]['idx']
        self.init_force[0]['pos'] = self.force[0]['pos']
        self.init_force[0]['radius'] = self.force[0]['radius']
        self.init_force[0]['strength'] = self.force[0]['strength']
        self.force[1]['idx'] = pyflex.add_forcefield()
        self.force[1]['pos'] = np.array(initPos2)
        self.force[1]['radius'] = radius
        self.force[1]['strength'] = strength
        self.init_force[1]['idx'] = self.force[1]['idx']
        self.init_force[1]['pos'] = self.force[1]['pos']
        self.init_force[1]['radius'] = self.force[1]['radius']
        self.init_force[1]['strength'] = self.force[1]['strength']
        print("forces: {}".format(self.force))
        # pyflex.add_sphere(0.01, initPos1, [1, 0, 0, 0])
        # pyflex.add_sphere(0.01, initPos2, [1, 0, 0, 0])
        # sst = pyflex.get_shape_states()
        # print("init states: {}".format(sst))
        # print("force  0  pos: {} from {}".format(self.force[0]['pos'], initPos1))

    def addSticky(self, radius=0.1, initPos=[-0.6, 0.2, -0.8]):
        self.init_sticky = initPos[0:3]
        self.prev_sticky = initPos[0:3]
        self.sticky_dist = radius
        print("sticky starts at: {}".format(self.prev_sticky))

    def get_sticky(self):
        return self.prev_sticky

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
            # new_dist = min(new_dist, 0.6)
            new_dist = max(new_dist, 0)  # 0.11)
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
        # print("prev dists: {}".format(self.prev_dist))
        if self.checkSphereCollisions(cur_pos[0][0:3], cur_pos[1][0:3], cur_pos[2][0:3], cur_pos[3][0:3]):
            pyflex.set_shape_states(cur_pos.flatten())
            self.prev_middle = cur_middle
            self.prev_middle[0, 1] = max(self.prev_middle[0, 1], 0.01)
            self.prev_middle[1, 1] = max(self.prev_middle[1, 1], 0.01)
        else:
            self.prev_rot = self.prev_rot - action[:, 3]
            self.prev_dist = self.prev_dist - action[:, 4]

    def boxStep(self, action, last_pos):
        action = np.reshape(action, [-1, 5])
        cur_pos = np.array(pyflex.get_shape_states())
        cur_middle = np.zeros((2, 3))
        for i in range(0, 2):
            actionPos = action[i, 0:3]
            actionRot = action[i, 3]
            actionDist = action[i, 4]

            new_rot = actionRot + self.prev_rot[i]
            new_dist = actionDist + self.prev_dist[i]
            # new_dist = min(new_dist, 0.6)
            new_dist = max(new_dist, 0.11)
            last_pos = np.reshape(last_pos, [-1, 14])
            cur_pos = np.reshape(cur_pos, [-1, 14])
            # cur_middle = cur_pos[0][0:3] +  np.array([self.prev_dist[i]/2 * np.cos(new_rot), 0, self.prev_dist[i]/2 * np.sin(new_rot)])
            # print("cur middle: {} prev_mid: {}".format(cur_middle, self.prev_middle))
            cur_middle[i, :] = self.prev_middle[i, :] + actionPos
            offset = np.array([new_dist / 2 * np.cos(new_rot), 0.0, new_dist / 2 * np.sin(new_rot)])
            cur_pos[i * 2][0:3] = cur_middle[i, :] - offset
            cur_pos[i * 2 + 1][0:3] = cur_middle[i, :] + offset
            # cur_pos[i*2][0] = cur_pos[i*2][0] -0.005
            # cur_pos[i * 2][2] = cur_pos[i * 2][0] - 0.005
            cur_pos[i * 2][3:6] = last_pos[i * 2][0:3]
            cur_pos[i * 2 + 1][3:6] = last_pos[i * 2 + 1][0:3]
            rvec1 = R.as_quat(R.from_rotvec(np.array([0, -new_rot + np.pi / 2, 0])))
            rvec2 = R.as_quat(R.from_rotvec(np.array([0, -abs(new_rot), 0])))

            print("new rot quat: {}".format(rvec1))

            cur_pos[i * 2][6:10] = rvec1
            cur_pos[i * 2 + 1][6:10] = rvec1
            cur_pos[i * 2][10:14] = last_pos[i * 2][6:10]
            cur_pos[i * 2 + 1][10:14] = last_pos[i * 2][6:10]

            self.prev_rot[i] = new_rot
            self.prev_dist[i] = new_dist
        # print("prev dists: {}".format(self.prev_dist))
        if self.checkSphereCollisions(cur_pos[0][0:3], cur_pos[1][0:3], cur_pos[2][0:3], cur_pos[3][0:3]):
            pyflex.set_shape_states(cur_pos.flatten())
            self.prev_middle = cur_middle
            self.prev_middle[0, 1] = max(self.prev_middle[0, 1], 0.01)
            self.prev_middle[1, 1] = max(self.prev_middle[1, 1], 0.01)
        else:
            self.prev_rot = self.prev_rot - action[:, 3]
            self.prev_dist = self.prev_dist - action[:, 4]

    def forceStep(self, action):
        action = np.reshape(action, [-1, 5])
        # shapeState = pyflex.get_shape_states()
        # shapeState = np.reshape(shapeState, [-1, 14])
        print("force fields: {}".format(self.force))
        for i in range(0, 2):
            self.force[i]['pos'] = self.force[i]['pos'] + action[i][0:3]
            self.force[i]['pos'][1] = max(0.0, self.force[i]['pos'][1])
            self.force[i]['strength'] = self.force[i]['strength'] + action[i][3]
            self.force[i]['radius'] = self.force[i]['radius'] + action[i][4]
            # if action[i][3] > 0:
            #    self.force[i]['strength'] = 150.0
            # else:
            #    self.force[i]['strength'] = 0
            pyflex.set_forcefield(self.force[i]['idx'], self.force[i]['pos'], self.force[i]['radius'], -self.force[i]['strength'])

            # if action[i][3] < 1:
            #    print('making sphere {} sticky at {}'.format(i, shapeState[i][0:3]))
            #    shapeState[i][3:6] = shapeState[i][0:3]
            # shapeState[i][0:3] = self.force[i]['pos']
            # shapeState[i][0:3] = shapeState[i][0:3]+action[i][0:3]
            # shapeState[i][1] = max(0, shapeState[i][1])
        # pyflex.set_shape_states(shapeState)

    """
    action[0:3] = delta on sticky point pos
    action[3] = stickiness
    """

    def sticky_step(self, action):
        action[3] = 0.1
        print("stickiness {}".format(action[3]))
        if action[3] < 0:
            self.sticky_idx = None
        elif self.sticky_idx == None:
            points = np.reshape(np.array(pyflex.get_positions()), [-1, 4])
            points = points[:, 0:3]
            diffs = np.subtract(points, self.prev_sticky)
            dists = np.sum(np.abs(diffs) ** 2, axis=1)
            minidx = np.argmin(dists)
            minVal = dists[minidx]
            self.sticky_pos = points[minidx, :]
            if minVal <= self.sticky_dist:
                self.sticky_idx = minidx
        else:
            points = np.reshape(np.array(pyflex.get_positions()), [-1, 4])
            print("moving point at {}".format(points[self.sticky_idx, 0:3]))
            self.sticky_pos = self.sticky_pos + action[0:3]
            points[self.sticky_idx, 0:3] = self.sticky_pos
            # points[self.sticky_idx, 0:3] = points[self.sticky_idx, 0:3] + action[0:3]
            print("To {}".format(points[self.sticky_idx, 0:3]))

            pyflex.set_positions(points)
        self.prev_sticky = self.prev_sticky + action[0:3]

    def sphere_reset(self):
        self.prev_middle = self.init_mid
        self.prev_rot = self.init_rot
        self.prev_dist = self.init_dist
        # print("prev_mid reset: {}".format(self.prev_middle))

    def box_reset(self):
        self.prev_middle = self.init_mid
        self.prev_rot = self.init_rot
        self.prev_dist = self.init_dist
        print("prev_mid reset: {}".format(self.prev_middle))

    def force_reset(self):
        for i in range(0, 2):
            print("#{}: init force pos: {} current force pos: {} {}".format(i, self.force[i]['pos'], self.init_force[i]['pos'],
                                                                            self.force[i]['strength']))
            self.force[i]['pos'] = self.init_force[i]['pos']
            self.force[i]['radius'] = self.init_force[i]['radius']
            self.force[i]['strength'] = self.init_force[i]['strength']

    def sticky_reset(self):
        self.prev_sticky[0] = self.init_sticky[0]
        self.prev_sticky[1] = self.init_sticky[1]
        self.prev_sticky[2] = self.init_sticky[2]
        self.sticky_idx = None

    def get_state(self):
        cur_state = super().get_state()
        cur_state['middle'] = self.prev_middle
        cur_state['rot'] = self.prev_rot
        cur_state['dist'] = self.prev_dist
        cur_state['force'] = self.force
        return cur_state

    def set_state(self, state_dict):
        self.prev_middle = state_dict['middle']
        self.prev_dist = state_dict['dist']
        self.prev_rot = state_dict['rot']
        self.force = state_dict['force']
        super().set_state(state_dict)

    def checkSphereCollisions(self, sp1, sp2, sp3, sp4):
        sp13dist = np.linalg.norm(sp1 - sp3)
        sp14dist = np.linalg.norm(sp1 - sp4)
        sp23dist = np.linalg.norm((sp2 - sp3))
        sp24dist = np.linalg.norm(sp2 - sp4)

        return sp13dist > self.radii and sp14dist > self.radii and sp23dist > self.radii and sp24dist > self.radii
