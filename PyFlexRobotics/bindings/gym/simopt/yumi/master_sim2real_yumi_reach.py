#!/usr/bin/env python
import argparse, logging, os, time, pickle, copy
import zmq
import numpy as np
import tensorflow as tf
from autolab_core import YamlConfig
from shutil import copyfile
from reps import Reps
from baselines import bench, logger
from baselines.common import set_global_seeds, boolean_flag, tf_util as U
from baselines.ppo1 import pposgd_vecenv

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env
from baselines.ppo1.mlp_policy import MlpPolicy, MlpBetaPolicy, MlpRaycastPolicy, MlpRaycastCNNPolicy

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
    input("Robot going to " + str(desired_pos) + ". Enter to start...")

    p_gain = 70.0  # 0.7
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


def run_robot(pi, egm_robot, total_timesteps, max_ep_len, obj_track_fn,
              init_position, stochastic = False):
    input("Enter any key to run real robot...")

    control_iter = int(robot_frequency / control_frequency)
    num_obs = 7 + 3
    obs = np.zeros((total_timesteps, num_obs))
    acs = np.zeros((total_timesteps, 8))
    vpreds = np.zeros((total_timesteps, 1))
    news = np.zeros((total_timesteps, 1), 'int32')

    ac_robot = np.zeros(7)
    ob = np.zeros(num_obs)
    prev_ob = np.zeros(num_obs)

    control_t = 0
    t = 0.0
    ac_gain = 0.3  # 0.15
    ramp_t = 50.0
    curr_target, _ = egm_robot.get_joint_state()
    curr_target = np.asarray(curr_target)

    while True:
        ramp_mult = min(t / ramp_t, 1.0)
        curr_state, _ = egm_robot.get_joint_state()
        curr_state = np.asarray(curr_state)

        if t % control_iter == 0:
            if control_t == total_timesteps:
                return {"ob": obs.copy(), "vpred": vpreds.copy(), "ac": acs.copy(),
                        "new": news.copy()}
            elif control_t % (max_ep_len + 1) == 0:
                print ("Current state..." + str(curr_state))

                print("Real. Timestep " + str(control_t) + ". Resetting...")
                curr_state = robot_go(egm_robot, init_position, curr_target)
                curr_target = curr_state
                news[control_t] = 1
                input("Real. Continuing with the next roll-out...")

            ob[:7] = curr_state
            ob[7:10] = obj_track_fn()

            # ob[7:10] = target_track_fn()
            # ob[10:13] = obj_track_fn()
            # ob[13:16] = prev_ob[10:13]
            prev_ob = ob
            ac, vpred = pi.act(stochastic, ob)
            # print (str(control_t) + " OB " + str(ob) + " --> AC " + str(ac))
            ac_robot = ac[:7]
            obs[control_t, :] = ob
            vpreds[control_t, :] = vpred
            acs[control_t, :] = ac

            control_t += 1

        curr_target = curr_target + ac_robot * ac_gain * ramp_mult * dt
        egm_robot.send_joint_command(curr_target)
        time.sleep(dt)
        t += 1


def run_sim(pi, env, horizon, stochastic):
    t = 0
    nenvs = env.num_envs
    ac = env.action_space.sample()  # not used, just so we have the datatype
    ac = np.repeat(np.expand_dims(ac, 0), nenvs, axis=0)
    new = [True for ne in range(nenvs)]  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = np.zeros(nenvs)  # return in current episode
    cur_ep_len = np.zeros(nenvs)  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.zeros((nenvs, horizon, ob.shape[1]), 'float32')
    rews = np.zeros((nenvs, horizon), 'float32')
    vpreds = np.zeros((nenvs, horizon), 'float32')
    news = np.zeros((nenvs, horizon), 'int32')
    acs = np.zeros((nenvs, horizon, ac.shape[1]), 'float32')
    prevacs = copy.deepcopy(acs)

    while True:
        prevac = ac

        ob = env.reset()
        ac, vpred = pi.act_parallel(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        sub_t = t % horizon
        if t > 0 and sub_t == 0:
            return {"ob": obs.copy(), "rew": rews.copy(), "vpred": vpreds.copy(), "new": news.copy(),
                   "ac": acs.copy(), "prevac": prevacs.copy(), "nextvpred": (vpred * (1 - new)).copy(),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy

        obs[:, sub_t] = ob
        vpreds[:, sub_t] = vpred
        news[:, sub_t] = new
        acs[:, sub_t] = ac
        prevacs[:, sub_t] = prevac

        # If using the beta policy, rescale the bounds to action_space bounds
        if type(pi.pd).__name__ == "BetaPd":
            ac = env.action_space.low + ac.copy() * (env.action_space.high - env.action_space.low)

        ob, rew, new, _ = env.step(ac)
        rews[:, sub_t] = rew
        cur_ep_ret += rew
        cur_ep_len += 1
        new_ids = np.where(new == 1)
        ep_rets.extend(cur_ep_ret[new_ids].tolist())
        ep_lens.extend(cur_ep_len[new_ids].tolist())
        cur_ep_ret[new_ids] = 0
        cur_ep_len[new_ids] = 0
        t += 1


def load_policy(env, policy_func, policy_path):
    print ("Loading policy from " + policy_path)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), policy_path)
    print ("Policy loaded!")
    return pi


def get_egm_robot():
    egm_address_right = ("", 6520)
    egm_address_left = ("", 6510)
    egm_robot = egm_bridge.EgmRobot("right", egm_address_right)
    return egm_robot


def run(num_iter, start_iter, rl_start_iter, cfg_env_source, cfg_env_source_init,
        cfg_rl, cfg_simopt, rl_logdir, real_data_path,
        iter_flag, run_rl_flag, run_simopt_flag, obj_track_fn):

    cwd_orig = os.getcwd()
    set_global_seeds(cfg_rl['seed'])
    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))

    def policy_fn(name, ob_space, ac_space):
        try:
            policy_class = eval(cfg_rl['policy_type'])
        except:
            logging.error('Policy type {} not found!'.format(cfg_rl['policy_type']))
            exit()
        return policy_class(name=name, ob_space=ob_space, ac_space=ac_space, **cfg_rl['policy'])

    # Copy the initial simulation parameters for the first iteration.
    copyfile(cfg_env_source_init, cfg_env_source + '_0.yaml')
    for i in range(start_iter, num_iter):
        sess = U.single_threaded_session()
        sess.__enter__()
        print("==== Starting iteration " + str(i) + " ====")
        # Write the current iteration index.
        iter_f = open(iter_flag, 'w+')
        iter_f.write(str(i))
        iter_f.close()

        if i >= rl_start_iter:
            # Start RL by creating the flag file.
            print("===> Run RL. Iteration " + str(i) + "...")
            run_rl_f = open(run_rl_flag, 'w+')
            run_rl_f.close()

            # Wait for RL training to finish.
            while True:
                if not os.path.isfile(run_rl_flag):
                    break
                time.sleep(1.0)

        # Collect data from the target sim (analogously to real)
        print("===> Collect real world data. Iteration " + str(i) + "...")
        input("Enter any key to run sim...")
        real_data_iter_path = os.path.realpath(real_data_path + "_" + str(i) + ".pkl")
        policy_path = os.path.realpath(os.path.join(rl_logdir + "_" + str(i),
                                   "{}-{}".format(cfg_rl['learn']['agent_name'],
                                                  cfg_rl['learn']['max_iters'])))

        cfg_env_path = cfg_env_source + "_" + str(i) + ".yaml"
        cfg_env = YamlConfig(cfg_env_path)
        cfg_env['gym']['renderBackend'] = 1
        cfg_env['scene']['SimParamsStochastic'] = False
        cfg_env['scene']['NumAgents'] = 1

        env = make_flex_vec_env(cfg_env)
        pi = load_policy(env, policy_fn, policy_path)
        seg_sim = run_sim(pi, env, cfg_simopt['learn']['timesteps_per_batch'], False)
        env.close()
        # egm_robot = get_egm_robot()
        # seg_real = run_robot(pi, egm_robot, cfg_simopt['learn']['timesteps_per_batch'], cfg_env['exp']['max_ep_len'],
        #                       obj_track_fn, cfg_env['scene']['InitPosition'][:7], False)

        seg = seg_sim
        print("===> Saving real data to " + real_data_iter_path)
        print ("OB shape" + str(seg['ob'].shape))

        # print ("SEG REAL *** " + str(seg_real['ob']))
        os.chdir(cwd_orig)
        real_data_f = open(real_data_iter_path, 'wb')
        pickle.dump(seg, real_data_f)
        real_data_f.close()

        # Start optimization of simulation parameters.
        print ("===> Run SimOpt. Iteration " + str(i) + "...")
        run_simopt_f = open(run_simopt_flag, 'w+')
        run_simopt_f.close()

        # Wait for simulation optimization to finish.
        while True:
            if not os.path.isfile(run_simopt_flag):
                break
            time.sleep(1.0)

        # Clean up TF session
        sess.__exit__(None, None, None)
        sess.__del__()
        U._PLACEHOLDER_CACHE = {}
        tf.reset_default_graph()
        import gc
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_iter', type=int, default=100)

    parser.add_argument('--cfg_env_source', type=str, default='/sim2real/log/yumi_reach')
    parser.add_argument('--cfg_env_source_init', type=str, default='/sim2real/cfg/yumi_reach_init.yaml')
    parser.add_argument('--cfg_rl', type=str, default='/sim2real/cfg/ppo1.yaml')
    parser.add_argument('--cfg_simopt', type=str, default='/sim2real/cfg/simopt.yaml')

    parser.add_argument('--rl_logdir', type=str, default='/sim2real/log/ppo1')
    parser.add_argument('--real_data_path', type=str, default='/sim2real/log/real')

    parser.add_argument('--iter_flag', type=str, default='/sim2real/log/iter.flag')
    parser.add_argument('--run_rl_flag', type=str, default='/sim2real/log/run_rl.flag')
    parser.add_argument('--run_simopt_flag', type=str, default='/sim2real/log/run_simopt.flag')
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--rl_start_iter', type=int, default=0)

    args = parser.parse_args()
    args.cfg_env_source = args.work_dir + args.cfg_env_source
    args.cfg_env_source_init = args.work_dir + args.cfg_env_source_init
    args.cfg_rl = args.work_dir + args.cfg_rl
    args.cfg_simopt = args.work_dir + args.cfg_simopt
    args.rl_logdir = args.work_dir + args.rl_logdir
    args.real_data_path = args.work_dir + args.real_data_path
    args.iter_flag = args.work_dir + args.iter_flag
    args.run_rl_flag = args.work_dir + args.run_rl_flag
    args.run_simopt_flag = args.work_dir + args.run_simopt_flag

    cfg_rl = YamlConfig(args.cfg_rl)
    cfg_simopt = YamlConfig(args.cfg_simopt)

    run_rl_flag = os.path.realpath(args.run_rl_flag)
    run_simopt_flag = os.path.realpath(args.run_simopt_flag)
    iter_flag = os.path.realpath(args.iter_flag)

    input("Press any key to start...")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    port = 5555
    topicfilter = "track"
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    socket.connect("tcp://localhost:%s" % port)

    def obj_track_fn():
        msg = socket.recv_string()
        topic, msg_data = msg.split()
        msg_elems = msg_data.split(",")
        # print("RECEIVED ", msg_elems)
        translation = np.asarray([float(msg_elems[0]), float(msg_elems[1]), float(msg_elems[2])])
        # return [-0.2, 0.5, 0.7]
        return translation

    run(args.num_iter, args.start_iter, args.rl_start_iter, args.cfg_env_source, args.cfg_env_source_init,
        cfg_rl, cfg_simopt, args.rl_logdir, args.real_data_path,
        iter_flag, run_rl_flag, run_simopt_flag, obj_track_fn)

    # robot_go(get_egm_robot(),
    #          np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # robot_go(get_egm_robot(),
    #          np.asarray([-0.19461474, -0.25002274,  0.25875276,  0.09130503, -0.02663598, -0.07177854, -0.09978467]))
