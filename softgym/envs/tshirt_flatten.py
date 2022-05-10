import numpy as np
import random
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object
from scipy.spatial.transform import Rotation as R
import cv2
import pickle
import os


class TshirtFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='tshirt_flatten_init_states.pkl', cloth_type='tshirt-small', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        self.cloth_type = cloth_type
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.shorts_pkl_path = os.path.join(cur_path, '../cached_initial_states/shorts_flatten.pkl')
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            self.set_scene(config)
            self._set_to_flat()
            self.move_to_pos([0, 0.05, 0])
            for _ in range(10):
                pyflex.step()
                # img = self.get_image()
                # cv2.imshow("image", img)
                # cv2.waitKey()
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            # pos[:, :3] -= np.mean(pos, axis=0)[:3]
            # if self.action_mode in ['sawyer', 'franka']:  # Take care of the table in robot case
            #     pos[:, 1] = 0.57
            # else:
            #     pos[:, 1] = 0.005
            # pyflex.set_positions(pos.flatten())
            # pyflex.set_velocities(np.zeros_like(pos))
            # pyflex.step()

            num_particle = pos.shape[0]
            pickpoint = random.randint(0, num_particle - 1)
            curr_pos = pyflex.get_positions()
            original_inv_mass = curr_pos[pickpoint * 4 + 3]
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            # pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
            # pickpoint_pos[1] += np.random.random(1) * 0.5 + 0.5
            # pickpoint_pos[1] +=  0.3
            pyflex.set_positions(curr_pos)

            # Pick up the cloth and wait to stablize
            obs = self._get_obs()
            # cv2.imshow("obs", obs)
            # cv2.waitKey()
            for pick_idx in range(1):
                pickup_t = 20
                for _ in range(pickup_t):
                    curr_pos = pyflex.get_positions()
                    curr_vel = pyflex.get_velocities()
                    curr_pos[pickpoint * 4 + 1] += 0.2 / pickup_t
                    curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                    pyflex.set_positions(curr_pos)
                    pyflex.set_velocities(curr_vel)
                    pyflex.step()
                    obs = self._get_obs()
                    # cv2.imshow("pick up obs", obs)
                    # cv2.waitKey()

                # for j in range(0, max_wait_step):
                #     curr_pos = pyflex.get_positions()
                #     curr_vel = pyflex.get_velocities()
                #     curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                #     curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                #     pyflex.set_positions(curr_pos)
                #     pyflex.set_velocities(curr_vel)
                #     pyflex.step()
                #     obs = self._get_obs()
                #     cv2.imshow("obs", obs)
                #     cv2.waitKey()
                #     if np.alltrue(np.abs(curr_vel) < stable_vel_threshold) and j > 5:
                #         break

                # Drop the cloth and wait to stablize
                curr_pos = pyflex.get_positions()
                curr_pos[pickpoint * 4 + 3] = original_inv_mass
                pyflex.set_positions(curr_pos)
                for _ in range(max_wait_step):
                    obs = self._get_obs()
                    # cv2.imshow("wait obs", obs)
                    # cv2.waitKey()
                    pyflex.step()
                    curr_vel = pyflex.get_velocities()
                    if np.alltrue(curr_vel < stable_vel_threshold):
                        break

            center_object()

            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function

            if self.cloth_type == 'tshirt' or self.cloth_type == 'tshirt-small':  # Use first frame as the flattened for tshirt and use the manual state for shorts
                generated_configs[-1]['flatten_area'] = self._set_to_flat(pos=pos)  # Record the maximum flatten area
            elif self.cloth_type == 'shorts':
                generated_configs[-1]['flatten_area'] = self._set_to_flat()  # Record the maximum flatten area

            print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.2, cy])
        pyflex.step()
        self.init_covered_area = None
        info = self._get_info()
        self.init_covered_area = info['performance']
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('key_point'):
            # TODO ad action_repeat
            print('Need to add action repeat')
            raise NotImplementedError
            raise DeprecationWarning
            valid_idxs = np.array([0, 63, 31 * 64, 32 * 64 - 1])
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            pyflex.step()

            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            action = action.reshape([-1, 4])
            idxs = np.hstack(action[:, 0])
            updates = action[:, 1:]
            action = np.hstack([action, np.zeros([action.shape[0], 1])])
            vels = pyflex.get_velocities()
            cur_pos[:, 3] = 1
            if self.action_mode == 'key_point_pos':
                cur_pos[valid_idxs[idxs.astype(int)], :3] = last_pos[valid_idxs[idxs.astype(int)]][:, :3] + updates
                cur_pos[valid_idxs[idxs.astype(int)], 3] = 0
            else:
                vels = np.array(vels).reshape([-1, 3])
                vels[idxs.astype(int), :] = updates
            pyflex.set_positions(cur_pos.flatten())
            pyflex.set_velocities(vels.flatten())
        else:
            self.action_tool.step(action)
            if self.action_mode in ['sawyer', 'franka']:
                pyflex.step(self.action_tool.next_action)
            else:
                pyflex.step()
        return

    def _get_current_covered_area(self, pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

        # Method 2
        # grid_copy = np.zeros([100, 100])
        # for x_low, x_high, y_low, y_high in zip(slotted_x_low, slotted_x_high, slotted_y_low, slotted_y_high):
        #     grid_copy[x_low:x_high, y_low:y_high] = 1
        # assert np.allclose(grid_copy, grid)
        # return np.sum(grid_copy) * span[0] * span[1]

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area
        return r

    # @property
    # def performance_bound(self):
    #     dimx, dimy = self.current_config['ClothSize']
    #     max_area = dimx * self.cloth_particle_radius * dimy * self.cloth_particle_radius
    #     min_p = 0
    #     max_p = max_area
    #     return min_p, max_p

    def _get_info(self):
        # Duplicate of the compute reward function!
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        init_covered_area = curr_covered_area if self.init_covered_area is None else self.init_covered_area
        max_covered_area = self.get_current_config()['flatten_area']
        info = {
            'performance': curr_covered_area,
            'normalized_performance': (curr_covered_area - init_covered_area) / (max_covered_area - init_covered_area),
            'normalized_performance_2': (curr_covered_area) / (max_covered_area),
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker, dtype=np.int32) * -1  # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps

    def get_picked_particle_new_position(self):
        intermediate_picked_particle_new_pos = self.action_tool.intermediate_picked_particle_pos
        if len(intermediate_picked_particle_new_pos) > 0:
            return np.vstack(intermediate_picked_particle_new_pos)
        else:
            return []

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 3
        if self.cloth_type == 'tshirt':
            cloth_type = 0
        elif self.cloth_type == 'shorts':
            cloth_type = 1
        else:
            cloth_type = 2
        scene_params = np.concatenate(
            [config['pos'][:], [config['scale'], config['rot']], config['vel'][:], [config['stiff'], config['mass'], config['radius']],
             camera_params['pos'][:], camera_params['angle'][:], [camera_params['width'], camera_params['height']], [render_mode], [cloth_type]])
        if self.version == 2:
            robot_params = []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            print("set scene")
            pyflex.set_scene(env_idx, scene_params, 0)
            print("after set scene")

        self.rotate_particles([0, 0, -90])
        self.move_to_pos([0, 0.05, 0])
        for _ in range(50):
            # print("after move to pos step {}".format(_))
            pyflex.step()
            # obs = self._get_obs()
            # cv2.imshow("obs at after move to obs", obs)
            # cv2.waitKey()
        self.default_pos = pyflex.get_positions()

        if state is not None:
            self.set_state(state)

        self.current_config = deepcopy(config)

    def get_default_config(self):
        # cam_pos, cam_angle = np.array([0.0, 0.82, 0.00]), np.array([0, -np.pi/2., 0.])
        cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])

        config = {
            'pos': [0.01, 0.15, 0.01],
            'scale': -1,
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': 1.0,
            'mass': 0.5 / (40 * 40),
            'radius': self.cloth_particle_radius,  # / 1.8,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height},
                              'top_down_camera_full': {
                                  'pos': np.array([0, 0.35, 0]),
                                  'angle': np.array([0, -90 / 180 * np.pi, 0]),
                                  'width': self.camera_width,
                                  'height': self.camera_height
                              },
                              },
            'drop_height': 0.0,
            'cloth_type': 0

        }

        return config

    def rotate_particles(self, angle):
        r = R.from_euler('zyx', angle, degrees=True)
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()[:, :3]
        new_pos = r.apply(new_pos)
        new_pos = np.column_stack([new_pos, pos[:, 3]])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _set_to_flat(self, pos=None):
        if pos is None:
            pos = self.default_pos
        if self.cloth_type == 'shorts':
            with open(self.shorts_pkl_path, 'rb') as f:
                pos = pickle.load(f)
        pyflex.set_positions(pos)
        # if self.cloth_type != 'shorts':
        #     self.rotate_particles([0, 0, 90])
        pyflex.step()
        return self._get_current_covered_area(pos)

    def move_to_pos(self, new_pos):
        # TODO
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos[:, :3] -= center[:3]
        pos[:, :3] += np.asarray(new_pos)
        pyflex.set_positions(pos)


if __name__ == '__main__':
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS
    import copy
    import cv2


    def prepare_policy(env):
        print("preparing policy! ", flush=True)

        # move one of the picker to be under ground
        shape_states = pyflex.get_shape_states().reshape(-1, 14)
        shape_states[1, :3] = -1
        shape_states[1, 7:10] = -1

        # move another picker to be above the cloth
        pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pp = np.random.randint(len(pos))
        shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
        shape_states[0, 7:10] = pos[pp] + [0., 0.06, 0.]
        pyflex.set_shape_states(shape_states.flatten())


    env_name = 'TshirtFlatten'
    env_args = copy.deepcopy(env_arg_dict[env_name])
    env_args['render_mode'] = 'cloth'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 720
    env_args['camera_width'] = 720
    env_args['camera_name'] = 'default_camera'
    env_args['headless'] = False
    env_args['action_repeat'] = 1
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    env_args['cached_states_path'] = 'tshirt_flatten_init_states_small_2021_05_28_01_16.pkl'
    env_args['num_variations'] = 20
    env_args['use_cached_states'] = True
    env_args['save_cached_states'] = False
    env_args['cloth_type'] = 'tshirt-small'
    # pkl_path = './softgym/cached_initial_states/shorts_flatten.pkl'

    env = SOFTGYM_ENVS[env_name](**env_args)
    print("before reset")
    env.reset()
    print("after reset")
    env._set_to_flat()
    print("after reset")
    # env.move_to_pos([0, 0.1, 0])
    # pyflex.step()
    # i = 0
    # import pickle

    # while (1):
    #     pyflex.step(render=True)
    #     if i % 500 == 0:
    #         print('saving pkl to ' + pkl_path)
    #         pos = pyflex.get_positions()
    #         with open(pkl_path, 'wb') as f:
    #             pickle.dump(pos, f)
    #     i += 1
    #     print(i)

    obs = env._get_obs()
    cv2.imwrite('./small_tshirt.png', obs)
    # cv2.imshow('obs', obs)
    # cv2.waitKey()

    prepare_policy(env)

    particle_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    n_particles = particle_positions.shape[0]
    # p_idx = np.random.randint(0, n_particles)
    # p_idx = 100
    pos = particle_positions
    ok = False
    while not ok:
        pp = np.random.randint(len(pos))
        if np.any(np.logical_and(np.logical_and(np.abs(pos[:, 0] - pos[pp][0]) < 0.00625, np.abs(pos[:, 2] - pos[pp][2]) < 0.00625),
                                 pos[:, 1] > pos[pp][1])):
            ok = False
        else:
            ok = True
    picker_pos = particle_positions[pp] + [0, 0.01, 0]

    timestep = 50
    movement = np.random.uniform(0, 1, size=(3)) * 0.4 / timestep
    movement = np.array([0.2, 0.2, 0.2]) / timestep
    action = np.zeros((timestep, 8))
    action[:, 3] = 1
    action[:, :3] = movement

    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 7:10] = -1

    shape_states[0, :3] = picker_pos
    shape_states[0, 7:10] = picker_pos

    pyflex.set_shape_states(shape_states)
    pyflex.step()

    obs_list = []

    for a in action:
        obs, _, _, _ = env.step(a)
        obs_list.append(obs)
        # cv2.imshow("move obs", obs)
        # cv2.waitKey()

    for t in range(30):
        a = np.zeros(8)
        obs, _, _, _ = env.step(a)
        obs_list.append(obs)
        # cv2.imshow("move obs", obs)
        # cv2.waitKey()

    from softgym.utils.visualization import save_numpy_as_gif

    save_numpy_as_gif(np.array(obs_list), '{}.gif'.format(
        env_args['cloth_type']
    ))
