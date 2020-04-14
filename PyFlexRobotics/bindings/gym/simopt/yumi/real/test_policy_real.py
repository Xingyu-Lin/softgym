#!/usr/bin/env python
import argparse, logging, os, time

import numpy as np
import tensorflow as tf
from autolab_core import YamlConfig
from baselines.common import set_global_seeds, boolean_flag, tf_util as U

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env
from baselines.ppo1.mlp_policy import MlpPolicy, MlpBetaPolicy, MlpRaycastPolicy, MlpRaycastCNNPolicy
import egm_bridge
import zmq

robot_frequency = 150.0
control_frequency = 20.0
dt = 1.0 / robot_frequency

joint_limits_low = np.asarray([-2.940880, -2.504547,  -2.155482, -5.061455, -1.535890, -3.996804, -2.940880])
joint_limits_high = np.asarray([2.940880, 0.759218, 1.396263, 5.061455, 2.408554, 3.996804, 2.940880])


def vel_control(curr, target, p_gain, dt, mult=1.0):
    vel_com = p_gain * (target - curr) * mult
    pos_com = curr + dt * vel_com
    pos_com = np.minimum(pos_com, joint_limits_high)
    pos_com = np.maximum(pos_com, joint_limits_low)
    return pos_com, vel_com


def robot_go(robot, desired_pos):
    p_gain = 100.0  # 0.7
    ramp_t = 100.0

    # Move real to init pose
    t = 0.0
    while True:
        ramp_mult = min(t / ramp_t, 1.0)
        curr_state, _ = robot.get_joint_state()
        curr_state = np.asarray(curr_state)
        pos_com, vel_com = vel_control(curr_state, desired_pos, p_gain, dt, ramp_mult)
        robot.send_joint_command(pos_com)
        if np.sum(np.abs(pos_com - desired_pos)) < 5e-3:
            break
        t += 1
        time.sleep(dt)


def robot_run_policy(pi, robot, timesteps, target_fn, stochastic = False):
    t = 0
    control_iter = int(robot_frequency / control_frequency)

    obs = np.zeros((timesteps, 10))
    acs = np.zeros((timesteps, 7))
    vpreds = np.zeros((timesteps, 1))
    ac = np.zeros(7)
    ob = np.zeros(10)

    curr_target, _ = robot.get_joint_state()

    control_t = 0
    t = 0.0
    ac_gain = 1.0

    ramp_t = 500.0
    while True:
        ob[:7], _ = robot.get_joint_state()
        ob[7:10] = target_fn()
        ramp_mult = min(t / ramp_t, 1.0)

        if t % control_iter == 0:
            ac, vpred = pi.act(stochastic, ob)
            # print (str(control_t) + " OB " + str(ob) +" --> AC " + str(ac))
            ac = ac[:7]
            obs[control_t, :] = ob
            vpreds[control_t, :] = vpred
            acs[control_t, :] = ac
            if control_t == timesteps:
                return {"ob": obs.copy(), "vpred": vpreds.copy(), "ac": acs.copy()}
            control_t += 1

        curr_target = curr_target + ac * dt * ramp_mult
        robot.send_joint_command(curr_target)
        time.sleep(dt)
        t += 1


def load_policy(env, policy_func, policy_path):
    print ("Loading policy from " + policy_path)
    ob_space = env.observation_space
    # ac_space = env.action_space

    from gym import spaces
    ac_space = spaces.Box(np.ones(8) * -1., np.ones(8) * 1.)

    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), policy_path)
    print ("Policy loaded!")
    return pi


def run(robot, cfg_env, cfg_rl, rl_logdir, rl_iter, timesteps, target_fn):

    cwd_orig = os.getcwd()

    sess = U.single_threaded_session()
    sess.__enter__()
    set_global_seeds(cfg_rl['seed'])

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../bin'))

    def policy_fn(name, ob_space, ac_space):
        try:
            policy_class = eval(cfg_rl['policy_type'])
        except:
            logging.error('Policy type {} not found!'.format(cfg_rl['policy_type']))
            exit()
        return policy_class(name=name, ob_space=ob_space, ac_space=ac_space, **cfg_rl['policy'])

    policy_path = os.path.realpath(os.path.join(rl_logdir,
                               "{}-{}".format(cfg_rl['learn']['agent_name'], rl_iter)))
    cfg_env['gym']['renderBackend'] = 0
    env = make_flex_vec_env(cfg_env)
    env.close()
    pi = load_policy(env, policy_fn, policy_path)

    init_position = cfg_env['scene']['InitPosition']
    print("Going to initial position: " + str(init_position))
    robot_go(robot, init_position)

    input("Press any key to run policy... ")
    target = np.asarray(cfg_env['scene']['TargetPosition'])
    seg = robot_run_policy(pi, robot, timesteps, target_fn, False)

    # Clean up TF session
    sess.__exit__(None, None, None)
    sess.__del__()
    U._PLACEHOLDER_CACHE = {}
    tf.reset_default_graph()
    import gc
    gc.collect()
    os.chdir(cwd_orig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_env', type=str, default='cfg/yumi_reach.yaml')
    parser.add_argument('--cfg_rl', type=str, default='cfg/train/ppo1.yaml')
    parser.add_argument('--rl_logdir', type=str, default='sim2real_yumi/policies/ppo1_yumi_reach_actrew')
    parser.add_argument('--rl_iter', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=1000)

    args = parser.parse_args()

    cfg_rl = YamlConfig(args.cfg_rl)
    cfg_env = YamlConfig(args.cfg_env)

    egm_address_right = ("", 6520)
    egm_address_left = ("", 6510)
    egm_robot = egm_bridge.EgmRobot("right", egm_address_right)
    input("Press any key to start...")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    port = 5555
    topicfilter = "track"
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    socket.connect("tcp://localhost:%s" % port)
    def target_fn():
        msg = socket.recv_string()
        topic, msg_data = msg.split()
        msg_elems = msg_data.split(",")
        # print("RECEIVED ", msg_elems)
        translation = np.asarray([float(msg_elems[0]), float(msg_elems[1]), float(msg_elems[2])])

        return translation


    run(egm_robot, cfg_env, cfg_rl, args.rl_logdir, args.rl_iter, args.timesteps, target_fn)