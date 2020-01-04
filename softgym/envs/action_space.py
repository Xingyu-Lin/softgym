import abc
import numpy as np
from gym.spaces import Box
import pyflex


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ Step funciton to change the action space states """


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

        space_low = np.array([-0.1, -0.1, -0.1, -0.1, -0.1] * 2) * 0.1  # [dx, dy, dz, dtheta, dh - open/close gripper]
        space_high = np.array([0.1, 0.1, 0.1, 0.1, 0.1] * 2) * 0.1
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
                center + np.array([diameter, 0., 0.]),
                center + np.array([diameter, 0., -diameter])]

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

    def step(self, action):
        pass
