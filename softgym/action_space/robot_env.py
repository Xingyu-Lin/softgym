import numpy as np
from softgym.action_space.action_space import ActionToolBase
from gym.spaces import Box
import pyflex


class RobotBase(ActionToolBase):
    def __init__(self, robot_name, control_method=None):
        super(RobotBase).__init__()
        assert robot_name in ['franka', 'sawyer']
        self.robot_name = robot_name
        self.control_method = control_method
        self.action_num = 8
        space_low = np.array([-0.01, -0.01, -0.01, -0.01])  # [dx, dy, dz, open/close gripper]
        space_high = np.array([0.01, 0.01, 0.01, 0.01])
        self.action_space = Box(space_low, space_high, dtype=np.float32)
        self.next_action = np.zeros(7,)

    def reset(self, state):
        for i in range(100):
            pyflex.step()

    def step(self, action):
        self.next_action = [*action[:3], 0., 0., 0., action[3]]
