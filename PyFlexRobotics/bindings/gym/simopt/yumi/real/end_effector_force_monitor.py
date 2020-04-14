# Copyright (c) 2018, NVIDIA  All rights reserved.

import argparse
import numpy as np
from numpy.linalg import inv, norm
import threading
import math
import sys
import rospy
from lula_control.status_subscriber_bundle import StatusSubscriberBundle
from lula_control.joint_states_listener import JointStatesListener, JointStatesView
from lula_control.matrix_listener import MatrixListener
from lula_pyutil import arg_parsing, util, math_util
from lula_yumi.yumi import YuMi

class JacobianColumn(object):
    def __init__(self, link_status, end_effector_status):
        self.link_status = link_status
        self.end_effector_status = end_effector_status
    
    def calc(self):
        ax = self.link_status.axis_z
        x = self.end_effector_status.orig
        o = self.link_status.orig

        v = np.cross(ax, x - o)
        return v

class YuMiJacobian(object):
    def __init__(self, side):
        self.side = side
        if self.side == 'right':
            suffix = 'r'
        elif self.side == 'left':
            suffix = 'l'
        else:
            raise RuntimeError('Invalid side: %s' % self.side)

        #self.yumi_joint_names= [name + suffix for name in [
        #        'yumi_joint_1_', 'yumi_joint_2_', 'yumi_joint_7_', 
        #        'yumi_joint_3_', 
        #        'yumi_joint_4_', 'yumi_joint_5_', 'yumi_joint_6_']]

        self.link1_status = StatusSubscriberBundle(
                '/robot/yumi_link_1_' + suffix)
        self.link1_status.wait_until_available()
        self.link2_status = StatusSubscriberBundle(
                '/robot/yumi_link_2_' + suffix)
        self.link2_status.wait_until_available()
        self.link3_status = StatusSubscriberBundle(
                '/robot/yumi_link_3_' + suffix)
        self.link3_status.wait_until_available()
        self.link4_status = StatusSubscriberBundle(
                '/robot/yumi_link_4_' + suffix)
        self.link4_status.wait_until_available()
        self.link5_status = StatusSubscriberBundle(
                '/robot/yumi_link_5_' + suffix)
        self.link5_status.wait_until_available()
        self.link6_status = StatusSubscriberBundle(
                '/robot/yumi_link_6_' + suffix)
        self.link6_status.wait_until_available()
        self.link7_status = StatusSubscriberBundle(
                '/robot/yumi_link_7_' + suffix)
        self.link7_status.wait_until_available()

        self.end_effector_status = StatusSubscriberBundle(
                '/robot/' + self.side + '_gripper')
        self.end_effector_status.wait_until_available()

        self.J_joint1_column = JacobianColumn(self.link1_status, self.end_effector_status)
        self.J_joint2_column = JacobianColumn(self.link2_status, self.end_effector_status)
        self.J_joint3_column = JacobianColumn(self.link3_status, self.end_effector_status)
        self.J_joint4_column = JacobianColumn(self.link4_status, self.end_effector_status)
        self.J_joint5_column = JacobianColumn(self.link5_status, self.end_effector_status)
        self.J_joint6_column = JacobianColumn(self.link6_status, self.end_effector_status)
        self.J_joint7_column = JacobianColumn(self.link7_status, self.end_effector_status)

    def calc(self):
        v1 = self.J_joint1_column.calc()
        v2 = self.J_joint2_column.calc()
        v3 = self.J_joint3_column.calc()
        v4 = self.J_joint4_column.calc()
        v5 = self.J_joint5_column.calc()
        v6 = self.J_joint6_column.calc()
        v7 = self.J_joint7_column.calc()
        J = np.zeros((3,7))
        J[:,0] = v1
        J[:,1] = v2
        J[:,2] = v3
        J[:,3] = v4
        J[:,4] = v5
        J[:,5] = v6
        J[:,6] = v7
        return J

class ExpAvgCollector(object):
    # Decay Exponential weighting over exp_weight_decay_duration_sec seconds
    # under the expected refresh rate of refresh_hz.
    def __init__(self, 
            exp_weight_decay_duration_sec=1., 
            decay_to_value=.04, 
            refresh_hz=30.):
        self.alpha = math.pow(
                decay_to_value, 
                1./(exp_weight_decay_duration_sec * refresh_hz))
        self.avg_vec = None

    def consume(self, vec):
        if self.avg_vec is None:
            self.avg_vec = vec
        else:
            self.avg_vec *= self.alpha
            self.avg_vec += (1. - self.alpha) * vec


class EndEffectorForceMonitor(object):
    def __init__(self, side, end_effector_status = None, use_joint_scaling=False):
        self.use_joint_scaling = use_joint_scaling
        self.side = side
        if side == 'right':
            suffix = 'r'
        elif side == 'left':
            suffix = 'l'
        else:
            raise RuntimeError('Unrecognized side: ' + side)

        self.ros_joints = [name + suffix for name in [
                'yumi_joint_1_', 'yumi_joint_2_', 'yumi_joint_7_', 
                'yumi_joint_3_', 
                'yumi_joint_4_', 'yumi_joint_5_', 'yumi_joint_6_']]
        self.joint_states = JointStatesView(
                JointStatesListener('/robot/joint_efforts'),
                self.ros_joints)
        print 'Waiting for joint states...'
        self.joint_states.wait_for_efforts()
        self.jacobian_listener = MatrixListener(
                '/robot/kinematics/jacobians/' + self.side + '/' + self.side + '_gripper/')
        print 'Waiting for jacobian...'
        self.jacobian_listener.wait_until_available()
        print '<done>'

        self.end_effector_status = end_effector_status

        self.avg_effort = ExpAvgCollector(1., .04, 30.)

        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self.run_monitor)
        self.monitor_thread.start()
        print 'Running...'

    def run_monitor(self):
        rate = rospy.Rate(30.)
        while not rospy.is_shutdown():
            try:
                self.lock.acquire()
                # TODO: Finish this.
                #efforts = self.joint_states.get_efforts()
                f = self._raw_external_force()
                self.avg_effort.consume(f)
            finally:
                self.lock.release()
            rate.sleep()

    def efforts(self):
        try:
            self.lock.acquire()
            return self.joint_states.get_efforts()
        finally:
            self.lock.release()


    # Returns the current external force. If in_end_effector_coords,
    # returns the force in the coordinate system of the end-effector.
    def _raw_external_force(self, in_end_effector_coords = False):
        efforts = self.joint_states.get_efforts()
        if self.use_joint_scaling:
            efforts[0] *= .2
            efforts[1] *= .3
            efforts[2] *= .4
            efforts[3] *= .7
            efforts[4] *= .7
            efforts[5] *= 1.
            efforts[6] *= 1.
        if efforts is None: return None
            
        J = self.jacobian_listener.matrix
        f = np.dot(np.dot(inv(np.dot(J, J.transpose())), J), efforts)
        #f = np.dot(J, efforts)

        if in_end_effector_coords:
            if not self.end_effector_status:
                raise RuntimeError(
                        'End-effector force requested in end-effector coordinates, but external force listener not provided an end-effector status monitor on initialization.')
            T_eff2base = self.end_effector_status.transform_matrix
            f_aug = np.ones(4)
            f_aug[0:3] = f
            f_eff = np.dot(inv(T_eff2base), f_aug)[0:3]
            f = f_eff

        return f

    # The public version is atomic.
    def raw_external_force(self, *args, **kwargs):
        try:
            self.lock.acquire()
            return self._raw_external_force(*args, **kwargs)
        finally:
            self.lock.release()

    def centered_external_force(self):
        try:
            self.lock.acquire()
            avg_f = self.avg_effort.avg_vec
            f = self._raw_external_force()
            if f is None or avg_f is None:
                return None
            return f - avg_f
        finally:
            self.lock.release()


