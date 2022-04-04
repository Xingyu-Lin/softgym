import numpy as np
from gym import spaces
import pyflex
from softgym.envs.cloth_env import FlexEnv
from softgym.action_space.action_space import PickerPickPlace
from softgym.utils.gemo_utils import *
from copy import deepcopy
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

def uv_to_world_pos(camera_params, depth, u, v, particle_radius=0.0075, on_table=False):
    height, width = depth.shape
    K = intrinsic_from_fov(height, width, 45) # the fov is 90 degrees

    # from cam coord to world coord
    cam_x, cam_y, cam_z = camera_params['default_camera']['pos'][0], camera_params['default_camera']['pos'][1], camera_params['default_camera']['pos'][2]
    cam_x_angle, cam_y_angle, cam_z_angle = camera_params['default_camera']['angle'][0], camera_params['default_camera']['angle'][1], camera_params['default_camera']['angle'][2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0]) 
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z
    matrix = np.linalg.inv(rotation_matrix @ translation_matrix)

    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    z = depth[int(np.rint(u)), int(np.rint(v))]
    if on_table or z == 0:
        vec = ((v - x0) / fx, (u - y0) / fy)
        z = (particle_radius - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])
    else:
        # adjust for particle radius from depth image
        z -= particle_radius
        
    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    
    cam_coord = np.ones(4)
    cam_coord[:3] = (x, y, z)
    world_coord = matrix @ cam_coord

    return world_coord

class BimanualEnv(FlexEnv):
    def __init__(self, use_depth=False, particle_radius=0.00625, picker_radius=0.005, shape='default', **kwargs):
        self.cloth_particle_radius = particle_radius
        self.image_height = 200
        self.image_width = 200
        super().__init__(**kwargs)

        # cloth shape
        if shape == 'default':
            self.config = self.get_default_config()
        elif shape == 'rect':
            self.config = self.get_rect_config()
        else:
            raise Exception("invalid cloth shape")
            
        self.update_camera(self.config['camera_name'], self.config['camera_params'][self.config['camera_name']])
        self.action_tool = PickerPickPlace(num_picker=2, picker_radius=picker_radius, particle_radius=particle_radius, env=self,
                                               picker_low=(-0.6, 0.0125, -0.6), picker_high=(0.6, 0.6, 0.6), collect_steps=False)
        self.action_tool.delta_move = 0.005 # default: 0.005
        self.num_rotations = 8
        self.num_scales = 3
        self.REW_SCALE = 1
        self.resize_scales =[(2, 0.5), (1, 1), (0.5, 2)]
        self.MID_FOLD_DIST = 78
        self.action_distances = [0.065, 0.13, 0.26]
        self.action_directions = [0., np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi] 
        self.goal = None
        self.reset_act = np.array([0.,0.1,-0.6,0.,0.,0.1,-0.6,0.])
        self.reset_pos = np.array([0.,0.1,-0.6,0.,0.1,-0.6])

        self.use_depth = use_depth # depth obs

    def get_default_config(self):
        particle_radius = self.cloth_particle_radius
        cam_pos, cam_angle = np.array([-0.0, 0.65, 0.0]), np.array([0, -np.pi/2., 0.])
        config = {
            'ClothPos': [-0.15, 0.0, -0.15],
            'ClothSize': [int(0.30 / particle_radius), int(0.30 / particle_radius)],
            'ClothStiff': [2.0, 0.5, 1.0],  # Stretch, Bend and Shear #0.8, 1., 0.9 #1.0, 1.3, 0.9
            'mass': 0.0054,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': 720,
                                   'height': 720}},
            'flip_mesh': 0,
            'drop_height': 0.0
        }

        return config

    def get_rect_config(self):
        particle_radius = self.cloth_particle_radius
        cam_pos, cam_angle = np.array([-0.0, 0.65, 0.0]), np.array([0, -np.pi/2., 0.])
        config = {
            'ClothPos': [-0.15, 0.0, -0.10],
            'ClothSize': [int(0.30 / particle_radius), int(0.20 / particle_radius)],
            'ClothStiff': [2.0, 0.5, 1.0],  # Stretch, Bend and Shear #0.8, 1., 0.9 #1.0, 1.3, 0.9
            'mass': 0.0054,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': 720,
                                   'height': 720}},
            'flip_mesh': 0,
            'drop_height': 0.0
        }

        return config

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _get_flat_pos(self):
        dimx, dimy = self.config['ClothSize']
        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def set_picker_pos(self, picker_pos):
        picker_pos = np.reshape(picker_pos, [-1,3])
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        for i in range(shape_states.shape[0]):
            shape_states[i, 3:6] = picker_pos[i]
            shape_states[i, :3] = picker_pos[i]
        pyflex.set_shape_states(shape_states)

    def get_hsv_mask(self, img):
        img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(img_hsv, np.array([20, 50., 10.]), np.array([40, 255., 255.]))
        mask2 = cv2.inRange(img_hsv, np.array([80, 50., 10.]), np.array([100, 255., 255.]))
        mask = cv2.bitwise_or(mask1,mask2)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((2,2),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_rgbd(self):
        # rgbd = pyflex.render_sensor()
        rgb, depth = pyflex.render()
        rgb = np.array(rgb).reshape(self.camera_params['default_camera']['height'], self.camera_params['default_camera']['width'], 4)
        rgb = rgb[::-1, :, :]
        rgb = rgb[:, :, :3]
        depth = np.array(depth).reshape(self.camera_params['default_camera']['height'], self.camera_params['default_camera']['width'])
        # depth = rgbd[:, :, 3]
        return rgb, depth

    def _get_obs(self):
        img = Image.fromarray(self.get_image(720, 720))
        resized = img.resize(size=(200,200))
        resized = np.array(resized)
        mask = self.get_hsv_mask(resized) != 0
        resized[mask == False, :] = 0

        obs = {'color': resized, 'goal': self.goal}

        if self.use_depth:
            rgb, depth = self.get_rgbd()
            depth = depth*255
            depth = depth.astype(np.uint8)
            depth_st = np.dstack([depth, depth, depth])
            depth_img = Image.fromarray(depth_st)
            depth_img = depth_img.resize(size=(200,200))
            depth_img = np.array(depth_img)
            obs['depth'] = depth_img

        return obs

    def set_scene(self):
        render_mode = 2 # cloth
        env_idx = 0
        config = self.config
        camera_params = config['camera_params'][config['camera_name']]
        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], config['mass'], config['flip_mesh']],
                                 dtype=np.float32)
        pyflex.set_scene(env_idx, scene_params, 0)

    def reset(self, given_goal=None, given_goal_pos=None):
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
        obs = self._get_obs()

        return obs

    def _get_iou(self, img, goal):
        goal_mask = self.get_hsv_mask(goal)
        img_mask  = self.get_hsv_mask(img)
        inter = np.sum(np.bitwise_and(img_mask,goal_mask))
        union = np.sum(np.bitwise_or(img_mask,goal_mask))
        iou = (inter*1.0/(union+1e-10))
        return iou

    def compute_reward(self, goal_pos=None):
        if goal_pos is None:
            return 0
        start_pos = pyflex.get_positions().reshape(-1, 4)[:,:3]
        pos_metric = np.linalg.norm(goal_pos - start_pos, axis=1).mean()
        return pos_metric

    def step(self, action, record_continuous_video=False, img_size=None, pickplace=False, on_table=False):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        obs = self._get_obs()
        start_pos = pyflex.get_positions().reshape(-1, 4)[:,:3]
        for i in range(self.action_repeat):
            self._step(action, pickplace, on_table=on_table)
            if record_continuous_video and i % 2 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))
        nobs = self._get_obs()
        reward = self.compute_reward(self.goal_pos)
        info = self._get_info()

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        if record_continuous_video:
            info['flex_env_recorded_frames'] = frames
        return nobs, reward, done, info

    def _step(self, action, pickplace=False, on_table=True):
        rgb, depth = self.get_rgbd()
        # camera_params = self.camera_params
        # rgb, depth = pyflex.render()
        # rgb = np.array(rgb).reshape(camera_params['default_camera']['height'], camera_params['default_camera']['width'], 3)
        # rgb = rgb[::-1, :, :]
        # rgb = rgb[:, :, :3]
        # depth = rgbd[:, :, 3]
        if pickplace:
            pick_u1, pick_v1 = action[0]
            pick_u1 = int(np.rint(pick_u1/199 * 719))
            pick_v1 = int(np.rint(pick_v1/199 * 719))
            pick_pos_1 = uv_to_world_pos(self.camera_params, depth, pick_u1, pick_v1, particle_radius=self.cloth_particle_radius, on_table=on_table)[:3]

            place_u1, place_v1 = action[1]
            place_u1 = int(np.rint(place_u1/199 * 719))
            place_v1 = int(np.rint(place_v1/199 * 719))
            place_pos_1 = uv_to_world_pos(self.camera_params, depth, place_u1, place_v1, particle_radius=self.cloth_particle_radius, on_table=on_table)[:3]

            pick_u2, pick_v2 = action[2]
            pick_u2 = int(np.rint(pick_u2/199 * 719))
            pick_v2 = int(np.rint(pick_v2/199 * 719))
            pick_pos_2 = uv_to_world_pos(self.camera_params, depth, pick_u2, pick_v2, particle_radius=self.cloth_particle_radius, on_table=on_table)[:3]

            place_u2, place_v2 = action[3]
            place_u2 = int(np.rint(place_u2/199 * 719))
            place_v2 = int(np.rint(place_v2/199 * 719))
            place_pos_2 = uv_to_world_pos(self.camera_params, depth, place_u2, place_v2, particle_radius=self.cloth_particle_radius, on_table=on_table)[:3]
            # mid = (pick_pos + place_pos)/2 

            mid_1 = 0.075 # 0.075
            mid_2 = 0.075
        else: 
            u1, v1 = action[3], action[4]
            u2, v2 = action[5], action[6]
            # rescale from 200,200 to 720,720
            u1 = int(np.rint(u1/199 * 719))
            v1 = int(np.rint(v1/199 * 719))
            u2 = int(np.rint(u2/199 * 719))
            v2 = int(np.rint(v2/199 * 719))
            pick_pos_1 = uv_to_world_pos(self.camera_params, depth, u1, v1, particle_radius=self.cloth_particle_radius, on_table=on_table)[:3]
            pick_pos_2 = uv_to_world_pos(self.camera_params, depth, u2, v2, particle_radius=self.cloth_particle_radius, on_table=on_table)[:3]
            
            angle = self.action_directions[int(action[0])]
            dist1 = self.action_distances[int(action[1])]
            dist2 = self.action_distances[int(action[2])]
            
            move1 = np.array([dist1, 0, 0])
            rot1 = np.zeros(3)
            rot1[0] = (np.cos(angle) * move1[0] - np.sin(angle) * move1[2])
            rot1[2] = (np.sin(angle) * move1[0] + np.cos(angle) * move1[2])

            move2 = np.array([dist2, 0, 0])
            rot2 = np.zeros(3)
            rot2[0] = (np.cos(angle) * move2[0] - np.sin(angle) * move2[2])
            rot2[2] = (np.sin(angle) * move2[0] + np.cos(angle) * move2[2])

            place_pos_1 = pick_pos_1 + rot1
            place_pos_2 = pick_pos_2 + rot2

            #u1, v1 = action[2], action[3]
            if action[1] == 0:
                mid_1 = 0.075
            else:
                mid_1 = 0.1

            if action[2] == 0:
                mid_2 = 0.075
            else:
                mid_2 = 0.1

        over = np.array([pick_pos_1[0], 0.1, pick_pos_1[2], pick_pos_2[0], 0.1, pick_pos_2[2]])
        self.set_picker_pos(over)

        sub_actions = [
                       np.array([pick_pos_1[0],pick_pos_1[1],pick_pos_1[2],0, pick_pos_2[0],pick_pos_2[1],pick_pos_2[2],0,]),
                       np.array([pick_pos_1[0], mid_1, pick_pos_1[2], 1, pick_pos_2[0], mid_2, pick_pos_2[2], 1]),
                       np.array([place_pos_1[0], mid_1, place_pos_1[2], 1, place_pos_2[0], mid_2, place_pos_2[2], 1]),
                       np.array([place_pos_1[0], place_pos_1[1]+self.config['drop_height'], place_pos_1[2], 1, place_pos_2[0], place_pos_2[1]+self.config['drop_height'], place_pos_2[2], 1]),
                       np.array([place_pos_1[0], 0.1, place_pos_1[2], 0, place_pos_2[0], 0.1, place_pos_2[2], 0])]

        for sub in sub_actions:
            self.action_tool.step(sub)
            pyflex.step()

        # go to neutral position
        #self.action_tool.step(sub)
        self.set_picker_pos(self.reset_pos)
        for _ in range(20):
            self.action_tool.step(self.reset_act)
            pyflex.step()

        pyflex.step()

    def _get_info(self):
        return {}
