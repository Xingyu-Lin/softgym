#!/usr/bin/env python

# Copyright (c) 2015, Fetch Robotics Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Fetch Robotics Inc. nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL FETCH ROBOTICS INC. BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Kentaro Wada

import select
import sys
import termios
from threading import Lock
import tty

import actionlib
from control_msgs.msg import GripperCommandAction
from control_msgs.msg import GripperCommandGoal
import rospy
from sensor_msgs.msg import JointState


def getch():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class GripperKeyboard(object):

    OPEN_POSITION = 0.1
    CLOSED_POSITION = 0

    def __init__(self):
        self._lock = Lock()

        self.max_effort = 20.0
        self.position = None
        self._sub_pos = rospy.Subscriber('joint_states', JointState,
                                         self._set_state)

        self.action_name = 'gripper_controller/gripper_action'
        self.client = actionlib.SimpleActionClient(self.action_name,
                                                   GripperCommandAction)
        rospy.loginfo('Waiting for gripper_controller...')
        self.client.wait_for_server()
        rospy.loginfo('...connected.')

    def _set_state(self, joint_state):
        l_gripper_position = None
        r_gripper_position = None
        for joint, pos in zip(joint_state.name, joint_state.position):
            if joint == 'l_gripper_finger_joint':
                l_gripper_finger_pos = pos
            if joint == 'r_gripper_finger_joint':
                r_gripper_finger_pos = pos
        with self._lock:
            self.position = l_gripper_finger_pos + r_gripper_finger_pos

    def set_position(self, position):
        goal = GripperCommandGoal()
        goal.command.max_effort = self.max_effort
        goal.command.position = position
        # Fill in the goal here
        self.client.send_goal(goal)
        self.client.wait_for_result(rospy.Duration.from_sec(5.0))
        res = self.client.get_result()
        with self._lock:
            self.position = res.position

    def open(self):
        self.set_position(self.OPEN_POSITION)

    def close(self):
        self.set_position(self.CLOSED_POSITION)


if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('grippper_keyboard')

    gripper_keyboard = GripperKeyboard()

    gripper_bindings = {
        'o': 'Open gripper',
        'c': 'Close gripper',
        'E': 'Increase max_effort',
        'e': 'Decrease max_effort',
        's': 'Show status',
        '?': 'Show help',
    }
    usage = 'Usage: '
    usage += ''.join('\n  {}: {}'.format(k, v)
                       for k, v in gripper_bindings.items())
    usage += '\n  Ctrl-C to quit'

    try:
        print(usage)
        while True:
            c = getch()
            if c.lower() in gripper_bindings:
                if c.lower() == 'o':
                    gripper_keyboard.open()
                    rospy.loginfo('Opened gripper')
                elif c.lower() == 'c':
                    gripper_keyboard.close()
                    rospy.loginfo('Closed gripper')
                elif c.lower() == 's':
                    rospy.loginfo('''State:
  position: {}
  max_effort: {}'''.format(gripper_keyboard.position,
                           gripper_keyboard.max_effort))
                elif c == 'e':
                    gripper_keyboard.max_effort -= 1
                    rospy.loginfo('Decrease max_effort to: {}'
                          .format(gripper_keyboard.max_effort))
                elif c == 'E':
                    gripper_keyboard.max_effort += 1
                    rospy.loginfo('Increase max_effort to: {}'
                          .format(gripper_keyboard.max_effort))
                elif c == '?':
                    print(usage)
            else:
                if c == '\x03':
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
