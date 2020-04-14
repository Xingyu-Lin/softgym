#!/usr/bin/env python
#
# Copyright (c) 2018, NVIDIA.  All rights reserved.

import argparse
import socket
import select
import time
import numpy as np
import math
import sys
import threading
import yaml

from . import egm_pb2


class EgmRobot:
    # The side is just data. It isn't used in connections. The
    # local_receive_port is the local port this object will receive joint state
    # information on from robot's EGM. In EGM terminology, this is the 'sensor
    # port'.
    def __init__(self, side, local_bind_address):
        self.side = side
        # UDP socket server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.0)  # Equiv to setblocking(0) (nonblocking). Use select.

        # Local address (addr, port) to receive on. Often addr = '', which
        # means it receives on all addrs.
        print ('local bind address:', local_bind_address)
        try:
            self.sock.bind(local_bind_address)
        except socket.error:
            print('ERROR -- could not bind:' + str(socket.error))
            print('Something else has likely bound to that port. Please try another.')
            raise

        # Every time the socket is read from (using recvfrom()), the remote
        # address that the information is coming over is given. This member will
        # store the value of the latest address.
        self.latest_egm_address = None
        self.seqno = 1

    def is_ready(self):
        return self.latest_egm_address != None

    # Send message to robot
    def __send_message(self, sensor_msg, target_address = None):
        if not target_address:
            target_address = self.latest_egm_address

        if target_address:
            # Send to a particular remote address = (ip, port).
            bytes_sent = self.sock.sendto(
                    sensor_msg.SerializeToString(), 
                    target_address)
        else:
            raise RuntimeError('Sending a message before one is received')
   
    def __get_latest(self):
        # Get the message from the robot.
        # Check whether the socket has readable data. Works only if socket is bound.
        try:
            readable, _, _ = select.select([self.sock], [], [], .2)
            if len(readable) > 0:
                # We have data. Get it... recvfrom(<buffer_size>) reads from the bound socket.
                data, self.latest_egm_address = self.sock.recvfrom(8192)

                # ...and de-serialize it.
                robot = egm_pb2.EgmRobot()
                robot.ParseFromString(data)
                return robot
            else:
                print ('<no_data>')
                return None
        except select.error as err: # (errno, strerror):
            # print ('<select error>: %d, %s' % (errno, strerror))
            print ('<select error>: ', err)
            return None
        except socket.error as err: #(errno, strerror):
            # print ('<socket error>: %d, %s' % (errno, strerror))
            print ('<socket error>: ', err)
            return None


    def send_joint_command(self, command):
        sensor = egm_pb2.EgmSensor()
        hdr = egm_pb2.EgmHeader()
        hdr.seqno = self.seqno
        hdr.tm = int(time.time()/1000)
        # hdr.tm = (lula_command.t.secs * 1000 + lula_command.t.nsecs / 1000000)
        hdr.mtype = egm_pb2.EgmHeader.MSGTYPE_CORRECTION
        sensor.header.CopyFrom(hdr)

        planned = egm_pb2.EgmPlanned()
        planned.joints.joints[:] = []
        planned.externalJoints.joints[:] = []

        for joint in range(0,6):
            val = rad2deg(command[joint])
            planned.joints.joints.append(val)

        val = rad2deg(command[6])
        planned.externalJoints.joints.append(val)

        sensor.planned.CopyFrom(planned)
        self.seqno += 1
        self.__send_message(sensor)

    def get_joint_state(self):
        robot_data = self.__get_latest()
        curr_seqno = robot_data.header.seqno

        # new_data = not prev_seqno or prev_seqno != curr_seqno
        # prev_seqno = curr_seqno

        q = robot_data.feedBack.joints.joints
        qext = robot_data.feedBack.externalJoints.joints

        state = []
        for i in range(len(q)):
            state.append(q[i])
        for i in range(len(qext)):
            state.append(qext[i])

        return state, curr_seqno

def lerp(curr, target, inter_factor, mult=1.0):
    return curr + (target - curr) * inter_factor * mult

def max_vel_limit(curr, target, max_vel):
    sign = lambda a: (a > 0) - (a < 0)
    diff = target - curr

    vel = sign(diff) * min(abs(diff), max_vel)

    return curr + vel, vel

def deg2rad(val):
    return math.pi * val / 180.

def rad2deg(rad_val):
    return 180. * rad_val / math.pi

if __name__ == '__main__':
    egm_address_right = ("", 6520)
    egm_address_left = ("", 6510)

    side = "right"
    egm_robot = EgmRobot(side, egm_address_right)

    desired_joint_state = [-0.027727,-0.493535,0.702561, 0.563854, -1.278152, -1.174275, 0.528319]
    init_joint_state = [1.30,-2.00, 0.20,-1.00,0.750,-0.30, -0.75]

    #max_vel = [0.05] * 7

    desired_joint_state = init_joint_state

    inter_factor = 1.0

    t = 0.0
    ramp_t = 100.0
    while True:
        curr_state, curr_seqno = egm_robot.get_joint_state()
        print ("ST " + str(curr_state))
        print ("")
        command = []
        vels = []
        mult = min(t/ramp_t, 1.0)
        for i in range(len(desired_joint_state)):
            com = lerp(curr_state[i], desired_joint_state[i], inter_factor, mult)
            #com, vel = max_vel_limit(curr_state[i], desired_joint_state[i], max_vel[i])
            command.append(com)
            #vels.append(vel)

        print ("CM " + str(command) + " VEL " + str(vels) + " MULT " + str(mult))
        egm_robot.send_joint_command(command)
        t += 1
        time.sleep(0.01)