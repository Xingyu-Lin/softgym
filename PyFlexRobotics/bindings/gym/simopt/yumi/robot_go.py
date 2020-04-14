#!/usr/bin/env python
import argparse, logging, os, time, pickle, copy
import zmq
import numpy as np

import real.egm_bridge as egm_bridge

robot_frequency = 250.0
control_frequency = 30.0
dt = 1.0 / robot_frequency

joint_limits_low = np.asarray([-2.940880, -2.504547,  -2.155482, -5.061455, -1.535890, -3.996804, -2.940880])
joint_limits_high = np.asarray([2.940880, 0.759218, 1.396263, 5.061455, 2.408554, 3.996804, 2.940880])


def vel_control(curr, target, p_gain, dt, mult=1.0):
    vel_com = p_gain * (target - curr) * mult
    pos_com = curr + dt * vel_com
    pos_com = np.minimum(pos_com, joint_limits_high)
    pos_com = np.maximum(pos_com, joint_limits_low)
    return pos_com, vel_com


def robot_go(robot, desired_pos, curr_target=None):
    print("Robot going to " + str(desired_pos) + ". Enter to start...")

    p_gain = 40.0  # 0.7
    ramp_t = 1500.0

    if curr_target is None:
        curr_target, _ = robot.get_joint_state()
        curr_target = np.asarray(curr_target)

    # Move real to init pose
    t = 0.0
    while True:
        ramp_mult = min(t / ramp_t, 1.0)
        pos_com, vel_com = vel_control(curr_target, desired_pos, p_gain, dt, ramp_mult)
        curr_target = pos_com

        curr_state, _ = robot.get_joint_state()
        curr_state = np.asarray(curr_state)

        robot.send_joint_command(curr_target)
        if np.sum(np.abs(curr_state - desired_pos)) < 5e-3:
            break
        t += 1
        time.sleep(dt)
    return curr_state


def get_egm_robot():
    egm_address_right = ("", 6520)
    egm_address_left = ("", 6510)
    egm_robot = egm_bridge.EgmRobot("right", egm_address_right)
    return egm_robot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    input("Press any key to start...")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    port = 5555
    topicfilter = "track"
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    socket.connect("tcp://localhost:%s" % port)

    zmq_context = zmq.Context()

    gripper_port = 5559
    gripper_socket = zmq_context.socket(zmq.PUB)
    gripper_socket.bind("tcp://*:%s" % gripper_port)
    # Wait for the gripper socket to be ready.
    for _ in range(2):
        gripper_socket.send_string("gripper none,none")
        time.sleep(1.0)

    def gripper_fn(command):
        gripper_socket.send_string("gripper right," + command)

    while True:
        robot_go(get_egm_robot(),
                 np.asarray([0.0, -0.5, 0.5, 0.0, 0.5, 0.0, 0.0]))
        time.sleep(5.0)
    # robot_go(get_egm_robot(),
    #           np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # robot_go(get_egm_robot(),
    #          np.asarray([-0.19461474, -0.25002274,  0.25875276,  0.09130503, -0.02663598, -0.07177854, -0.09978467]))