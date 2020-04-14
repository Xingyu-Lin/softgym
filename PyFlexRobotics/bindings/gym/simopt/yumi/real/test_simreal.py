#!/usr/bin/env python
import argparse, os, time

import numpy as np
from autolab_core import YamlConfig

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env
import egm_bridge

joint_limits_low = np.asarray([-2.940880, -2.504547,  -2.155482, -5.061455, -1.535890, -3.996804, -2.940880])
joint_limits_high = np.asarray([2.940880, 0.759218, 1.396263, 5.061455, 2.408554, 3.996804, 2.940880])

def vel_control(curr, target, p_gain, dt, mult=1.0):

    vel_com = p_gain * (target - curr) * mult
    pos_com = curr + dt * vel_com

    pos_com = np.minimum(pos_com, joint_limits_high)
    pos_com = np.maximum(pos_com, joint_limits_low)

    return pos_com, vel_com


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_env', type=str, default='sim2real_yumi/cfg/yumi_cabinet_simreal.yaml')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim','real','simreal'])
    args = parser.parse_args()
    cfg_env = YamlConfig(args.cfg_env)
    mode = args.mode

    if mode == 'real' or mode == 'simreal':
        egm_address_right = ("", 6520)
        egm_address_left = ("", 6510)
        egm_robot = egm_bridge.EgmRobot("right", egm_address_right)
    if mode == 'sim' or mode == 'simreal':
        set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../bin'))
        env = make_flex_vec_env(cfg_env)
        ob = env.reset()

    dt = 1.0 / 250.0
    p_gain = 10 # 0.7
    ramp_t = 100.0

    desired_joint_state = np.asarray([-0.027727, -0.493535, 0.702561, 0.563854, -1.278152, -1.174275, 0.528319])
    init_joint_state = np.asarray([1.30, -2.00, 0.20, -1.00, 0.750, -0.30, -0.75])

    # Move real to init pose
    if mode == 'real' or mode == 'simreal':
        t = 0.0
        while True:
            ramp_mult = min(t / ramp_t, 1.0)
            curr_state, _ = egm_robot.get_joint_state()
            curr_state = np.asarray(curr_state)
            pos_com, vel_com = vel_control(curr_state, init_joint_state, p_gain, dt, ramp_mult)
            egm_robot.send_joint_command(pos_com)

            if np.sum(np.abs(pos_com - curr_state)) < 1e-4:
                break
            t += 1
            time.sleep(dt)

    # Move sim to init pose
    if mode == 'sim' or mode == 'simreal':
        while True:
            pos_com = np.zeros(8)
            pos_com[:7], vel_com = vel_control(ob[0,:7], init_joint_state, p_gain, dt, 1.0)
            ob, rew, new, _ = env.step(pos_com)

            if np.sum(np.abs(pos_com[:7] - ob[0,:7])) < 1e-4:
                break
            time.sleep(dt)

    resp = input("Press any key to start... ")

    # Move sim and real to the target pose
    t = 0.0
    while True:
        ramp_mult = min(t / ramp_t, 1.0)

        if mode == 'real' or mode == 'simreal':
            curr_state, _ = egm_robot.get_joint_state()
            pos_com, vel_com = vel_control(np.asarray(curr_state), desired_joint_state, p_gain, dt, ramp_mult)
            egm_robot.send_joint_command(pos_com)

        if mode == 'sim' or mode == 'simreal':
            pos_com = np.zeros(8)
            pos_com[:7], vel_com = vel_control(ob[0,:7], desired_joint_state, p_gain, dt, ramp_mult)
            ob, rew, new, _ = env.step(pos_com)
        t += 1
        time.sleep(dt)

