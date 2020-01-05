import abc
import numpy as np
from gym.spaces import Box
from softgym.utils.utils import rotation_2d_around_center, extend_along_center
import pyflex


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ Step funciton to change the action space states. Does not call pyflex.step() """


class ParallelGripper(ActionToolBase):
    # TODO add observation space
    def __init__(self, gripper_type='sphere', sphere_radius=0.1, init_pos=(0., -1., 0.)):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """
        super(ParallelGripper).__init__()

        # initial position of the spheres are near the pick point of the cloth
        # random_particle_position = pyflex.get_positions().reshape((-1, 4))[pick_point]
        # init_x = random_particle_position[0]
        # init_y = 0.5
        # init_z = random_particle_position[2]
        #
        # init_x += np.random.uniform(0, 0.1)
        # init_z += np.random.uniform(0, 0.1)

        # init_pos1 = [init_x, init_y, init_z]
        # init_pos2 = [init_x, init_y, init_z + 0.2]
        assert gripper_type in ['sphere', 'cube']
        if gripper_type == 'cube':
            raise NotImplementedError
        self.sphere_radius = sphere_radius

        gripper_poses = self._get_centered_gripper_pos(init_pos)
        for gripper_pos in gripper_poses:
            pyflex.add_sphere(sphere_radius, gripper_pos, [1, 0, 0, 0])

        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        space_low = np.array([-0.1, -0.1, -0.1, -0.4, -0.2] * 2) * 0.2  # [dx, dy, dz, dtheta, dh - open/close gripper]
        space_high = np.array([0.1, 0.1, 0.1, 0.4, 0.2] * 2) * 0.2
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    def _get_centered_gripper_pos(self, center):
        # -z
        # | left_2, right_2
        # | left_1, right_1
        # -----------------> x
        # return [left_1, left_2, right_1, right_2]
        diameter = 2 * self.sphere_radius
        return [center,
                center + np.array([0., 0., -diameter]),
                center + np.array([diameter + 0.2, 0., 0.]),
                center + np.array([diameter + 0.2, 0., -diameter])]

    def get_state(self):
        pass

    def set_state(self):
        pass

    def reset(self, center):
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_gripper_pos = self._get_centered_gripper_pos(center)
        for (i, centered_gripper_pos) in enumerate(centered_gripper_pos):
            shape_state[i] = np.hstack([centered_gripper_pos, centered_gripper_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        pyflex.step()

    @staticmethod
    def _get_current_gripper_pos():
        cur_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        return cur_pos[:, :3]

    @staticmethod
    def _set_gripper_pos(gripper_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = gripper_pos
        pyflex.set_shape_states(shape_states)

    def step(self, action):
        """ action = (translation, rotation, open/close gripper). First do transltation then rotation then gripper """
        action = np.reshape(action, [-1, 5])
        curr_gripper_pos = self._get_current_gripper_pos()
        next_gripper_pos = curr_gripper_pos.copy()

        for i in range(0, 2):
            action_pos = action[i, 0:3]
            action_rot = action[i, 3]
            action_dist = action[i, 4]
            curr_middle = np.mean(curr_gripper_pos[2 * i:2 * i + 2, :], axis=0)
            next_middle = curr_middle + action_pos
            next_gripper_pos[2 * i, :] = rotation_2d_around_center(curr_gripper_pos[2 * i, :], curr_middle, action_rot) + action_pos
            next_gripper_pos[2 * i + 1, :] = rotation_2d_around_center(curr_gripper_pos[2 * i + 1, :], curr_middle, action_rot) + action_pos
            next_gripper_pos[2 * i, :] = extend_along_center(next_gripper_pos[2 * i, :], next_middle, action_dist / 2., self.sphere_radius,
                                                             2 * self.sphere_radius)
            next_gripper_pos[2 * i + 1, :] = extend_along_center(next_gripper_pos[2 * i + 1, :], next_middle, action_dist / 2., self.sphere_radius,
                                                                 2 * self.sphere_radius)

        next_gripper_pos[:, 1] = np.maximum(next_gripper_pos[:, 1], self.sphere_radius)
        self._set_gripper_pos(next_gripper_pos)
        # if not self._check_sphere_collision(next_gripper_pos):
        #     self._set_gripper_pos(next_gripper_pos)

    def _check_sphere_collision(self, gripper_pos):
        """ Check if different grippers are in collision. Currently not used. """
        for i in range(gripper_pos.shape[0]):
            for j in range(i + 1, gripper_pos.shape[0]):
                if np.linalg.norm(gripper_pos[i, :] - gripper_pos[j, :]) < 2 * self.sphere_radius + 1e-2:
                    return True
        return False


class Picker(ActionToolBase):
    # TODO add observation space
    def __init__(self, gripper_type='sphere', sphere_radius=0.1, init_pos=(0., -1., 0.)):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """
        super(ParallelGripper).__init__()

        # initial position of the spheres are near the pick point of the cloth
        # random_particle_position = pyflex.get_positions().reshape((-1, 4))[pick_point]
        # init_x = random_particle_position[0]
        # init_y = 0.5
        # init_z = random_particle_position[2]
        #
        # init_x += np.random.uniform(0, 0.1)
        # init_z += np.random.uniform(0, 0.1)

        # init_pos1 = [init_x, init_y, init_z]
        # init_pos2 = [init_x, init_y, init_z + 0.2]
        assert gripper_type in ['sphere', 'cube']
        if gripper_type == 'cube':
            raise NotImplementedError
        self.sphere_radius = sphere_radius

        gripper_poses = self._get_centered_gripper_pos(init_pos)
        for gripper_pos in gripper_poses:
            pyflex.add_sphere(sphere_radius, gripper_pos, [1, 0, 0, 0])

        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        space_low = np.array([-0.1, -0.1, -0.1, -0.4, -0.2] * 2) * 0.2  # [dx, dy, dz, dtheta, dh - open/close gripper]
        space_high = np.array([0.1, 0.1, 0.1, 0.4, 0.2] * 2) * 0.2
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    def _get_centered_gripper_pos(self, center):
        # -z
        # | left_2, right_2
        # | left_1, right_1
        # -----------------> x
        # return [left_1, left_2, right_1, right_2]
        diameter = 2 * self.sphere_radius
        return [center,
                center + np.array([0., 0., -diameter]),
                center + np.array([diameter + 0.2, 0., 0.]),
                center + np.array([diameter + 0.2, 0., -diameter])]

    def get_state(self):
        pass

    def set_state(self):
        pass

    def reset(self, center):
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_gripper_pos = self._get_centered_gripper_pos(center)
        for (i, centered_gripper_pos) in enumerate(centered_gripper_pos):
            shape_state[i] = np.hstack([centered_gripper_pos, centered_gripper_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        pyflex.step()

    @staticmethod
    def _get_current_gripper_pos():
        cur_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        return cur_pos[:, :3]

    @staticmethod
    def _set_gripper_pos(gripper_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = gripper_pos
        pyflex.set_shape_states(shape_states)

    def step(self, action):
        """ action = (translation, rotation, open/close gripper). First do transltation then rotation then gripper """
        action = np.reshape(action, [-1, 5])
        curr_gripper_pos = self._get_current_gripper_pos()
        next_gripper_pos = curr_gripper_pos.copy()

        for i in range(0, 2):
            action_pos = action[i, 0:3]
            action_rot = action[i, 3]
            action_dist = action[i, 4]
            curr_middle = np.mean(curr_gripper_pos[2 * i:2 * i + 2, :], axis=0)
            next_middle = curr_middle + action_pos
            next_gripper_pos[2 * i, :] = rotation_2d_around_center(curr_gripper_pos[2 * i, :], curr_middle, action_rot) + action_pos
            next_gripper_pos[2 * i + 1, :] = rotation_2d_around_center(curr_gripper_pos[2 * i + 1, :], curr_middle, action_rot) + action_pos
            next_gripper_pos[2 * i, :] = extend_along_center(next_gripper_pos[2 * i, :], next_middle, action_dist / 2., self.sphere_radius,
                                                             2 * self.sphere_radius)
            next_gripper_pos[2 * i + 1, :] = extend_along_center(next_gripper_pos[2 * i + 1, :], next_middle, action_dist / 2., self.sphere_radius,
                                                                 2 * self.sphere_radius)

        next_gripper_pos[:, 1] = np.maximum(next_gripper_pos[:, 1], self.sphere_radius)
        self._set_gripper_pos(next_gripper_pos)
        # if not self._check_sphere_collision(next_gripper_pos):
        #     self._set_gripper_pos(next_gripper_pos)

    def _check_sphere_collision(self, gripper_pos):
        """ Check if different grippers are in collision. Currently not used. """
        for i in range(gripper_pos.shape[0]):
            for j in range(i + 1, gripper_pos.shape[0]):
                if np.linalg.norm(gripper_pos[i, :] - gripper_pos[j, :]) < 2 * self.sphere_radius + 1e-2:
                    return True
        return False
