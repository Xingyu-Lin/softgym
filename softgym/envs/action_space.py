import abc
import numpy as np
from gym.spaces import Box
from softgym.utils.utils import rotation_2d_around_center, extend_along_center
import pyflex
import scipy.spatial


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
    def __init__(self, num_picker=1, picker_radius=0.03, init_pos=(0., -0.1, 0.), picker_threshold=0.05):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [None] * self.num_picker

        init_picker_poses = self._get_centered_picker_pos(init_pos)
        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])

        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        space_low = np.array([-0.1, -0.1, -0.1, 0] * self.num_picker) * 0.2  # [dx, dy, dz, [0, 1]]
        space_high = np.array([0.1, 0.1, 0.1, 5] * self.num_picker) * 0.2
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.cos(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.sin(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack([centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        pyflex.step()

    @staticmethod
    def _get_pos():
        """ Get the current pos of the pickers and the particles, along with the inverse mass of each particle """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    def step(self, action):
        """ action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one, for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        action = np.reshape(action, [-1, 4])
        pick_flag = np.random.random(self.num_picker) < action[:, 3]
        picker_pos, particle_pos = self._get_pos()
        new_picker_pos, new_particle_pos = picker_pos.copy(), particle_pos.copy()

        # Un-pick the particles
        for i in range(self.num_picker):
            if not pick_flag[i] and self.picked_particles[i] is not None:
                new_particle_pos[self.picked_particles[i], 3] = 1.  # Revert the mass
                self.picked_particles[i] = None
        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]
            if pick_flag[i]:
                if self.picked_particles[i] is None:  # No particle is currently picked and thus need to select a particle to pick
                    dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                    idx_dists = np.hstack([np.arange(particle_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
                    mask = dists.flatten() <= self.picker_threshold + self.picker_radius  # TODO add the particle radius here too.
                    idx_dists = idx_dists[mask, :].reshape((-1, 2))
                    if idx_dists.shape[0] > 0:
                        pick_id, pick_dist = None, None
                        for j in range(idx_dists.shape[0]):
                            if idx_dists[j, 0] not in self.picked_particles and (pick_id is None or idx_dists[j, 1] < pick_dist):
                                pick_id = idx_dists[j, 0]
                                pick_dist = idx_dists[j, 1]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)

                if self.picked_particles[i] is not None:
                    # TODO The position of the particle needs to be updated such that it is close to the picker particle
                    new_particle_pos[self.picked_particles[i], :3] = particle_pos[self.picked_particles[i], :3] + action[i, :3]
                    new_particle_pos[self.picked_particles[i], 3] = 0  # Set the mass to infinity
        self._set_pos(new_picker_pos, new_particle_pos)
