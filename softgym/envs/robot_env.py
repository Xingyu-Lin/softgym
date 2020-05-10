from softgym.envs.action_space import ActionToolBase


class RobotBase(ActionToolBase):
    def __init__(self, robot_name, control_method = None):
        super(RobotBase).__init__()
        assert robot_name in ['franka', 'sawyer']
        self.robot_name = robot_name
        self.control_method = control_method

    def reset(self, state):
        pass

    def step(self, action):
        pass

    @property
    def action_space(self):
        """ Action space of the robot"""
        return None
