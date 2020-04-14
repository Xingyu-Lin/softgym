import rospy
import zmq
import numpy as np
from lula_yumi.abb_client import ABBClient
from lula_yumi.yumi import YuMi
from end_effector_force_monitor import *
from lula_control.joint_states_listener import JointStatesListener, JointStatesView
from sensor_msgs.msg import JointState


class LulaControlRepublisher:

    def __init__(self, sub_port=5560, pub_port=5561):
        rospy.init_node('lula_control_republisher')
        self.side = 'right'
        suffix = 'r'

        zmq_context = zmq.Context()
        topicfilter = u'lula'
        self.sub_socket = zmq_context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:%s" % sub_port)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
        # self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.end_effector_force_monitor = EndEffectorForceMonitor(self.side)

        zmq_context = zmq.Context()
        self.pub_socket = zmq_context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:%s" % pub_port)

        self.robot = YuMi(is_simulation=True)

        self.yumi_joint_names = [name + suffix for name in [
            'yumi_joint_1_', 'yumi_joint_2_', 'yumi_joint_7_',
            'yumi_joint_3_',
            'yumi_joint_4_', 'yumi_joint_5_', 'yumi_joint_6_']]

        self.robot_joints = JointStatesView(
            JointStatesListener('/robot/joint_states'),
            self.yumi_joint_names)
        self.robot_joints.wait_for_positions()
        self.curr_target = np.array(self.robot_joints.get_positions())

    def run(self):
        self.curr_target = np.array(self.robot_joints.get_positions())
        rate = rospy.Rate(30.0)
        print ("Waiting for control commands...")

        while not rospy.is_shutdown():
            try:
                msg = self.sub_socket.recv(flags=zmq.NOBLOCK)
                topic, msg_data = msg.split()
                print ("Command received: " + msg_data)
                msg_elems = msg_data.split(",")
                command = msg_elems[0]
                args = msg_elems[1:]

                if command == 'init_q':
                    self.curr_target = np.array(self.robot_joints.get_positions())
                elif command == 'qvel_c':
                    vel = np.asarray([float(arg) for arg in args])
                    self.go_vel(vel, compliant=True)
                elif command == 'qvel':
                    vel = np.asarray([float(arg) for arg in args])
                    self.go_vel(vel, compliant=False)
            except zmq.Again as e:
                pass

            msg_str = "%f,%f,%f,%f,%f,%f,%f" % tuple(self.robot_joints.get_positions())
            msg_str += ",%f,%f,%f,%f,%f,%f,%f" % tuple(self.curr_target)
            self.pub_socket.send_string("lula_state " + msg_str)
            rate.sleep()

    def go_vel(self, vel, compliant=False):
        self.curr_target += vel

        if compliant:
            efforts = np.array(self.end_effector_force_monitor.efforts())
            efforts[0] *= 1. * .05
            efforts[1] *= 1. * .05
            efforts[2] *= 1. * .05
            efforts[3] *= 1. * .15
            efforts[4] *= 1. * 0.7
            efforts[5] *= 1. * 0.7
            efforts[6] *= 1. * 0.7
            self.curr_target += 0.1 * efforts

        self.robot.side(self.side).end_effector.go_config(self.curr_target)


if __name__ == '__main__':
    lulacontrol_republisher = LulaControlRepublisher()
    lulacontrol_republisher.run()


