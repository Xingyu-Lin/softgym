#!/usr/bin/env python
import argparse, logging, os, time, pickle, copy

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


def run_target_sim(pi, env, horizon, stochastic):
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


def run(num_iter, cfg_env_source, cfg_env_source_init, cfg_env_target,
        cfg_rl, cfg_simopt, rl_logdir, real_data_path,
        iter_flag, run_rl_flag, run_simopt_flag):

    cwd_orig = os.getcwd()
    set_global_seeds(cfg_rl['seed'])

    cfg_env_target['gym']['seed'] = cfg_rl['seed']
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
    for i in range(0, num_iter):
        sess = U.single_threaded_session()
        sess.__enter__()
        print("==== Starting iteration " + str(i) + " ====")
        # Write the current iteration index.
        iter_f = open(iter_flag, 'w+')
        iter_f.write(str(i))
        iter_f.close()

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
        real_data_iter_path = os.path.realpath(real_data_path + "_" + str(i) + ".pkl")
        policy_path = os.path.realpath(os.path.join(rl_logdir + "_" + str(i),
                                   "{}-{}".format(cfg_rl['learn']['agent_name'],
                                                  cfg_rl['learn']['max_iters'])))
        env = make_flex_vec_env(cfg_env_target)
        pi = load_policy(env, policy_fn, policy_path)
        seg = run_target_sim(pi, env, cfg_simopt['learn']['timesteps_per_batch'], False)
        env.close()

        print("===> Saving real data to " + real_data_iter_path)
        print ("OB shape" + str(seg['ob'].shape))
        os.chdir(cwd_orig)
        real_data_f = open(real_data_iter_path, 'wb')
        pickle.dump(seg, real_data_f)
        # pickle.dump(seg['ac'], f)
        # pickle.dump(seg['new'], f)
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

    parser.add_argument('--cfg_env_source', type=str, default='/sim2real/log/yumi_ropepeg')
    parser.add_argument('--cfg_env_source_init', type=str, default='/sim2real/cfg/yumi_ropepeg_init.yaml')
    parser.add_argument('--cfg_env_target', type=str, default='/sim2real/cfg/yumi_ropepeg_target.yaml')
    parser.add_argument('--cfg_rl', type=str, default='/sim2real/cfg/ppo1.yaml')
    parser.add_argument('--cfg_simopt', type=str, default='/sim2real/cfg/simopt.yaml')

    parser.add_argument('--rl_logdir', type=str, default='/sim2real/log/ppo1')
    parser.add_argument('--real_data_path', type=str, default='/sim2real/log/real')

    parser.add_argument('--iter_flag', type=str, default='/sim2real/log/iter.flag')
    parser.add_argument('--run_rl_flag', type=str, default='/sim2real/log/run_rl.flag')
    parser.add_argument('--run_simopt_flag', type=str, default='/sim2real/log/run_simopt.flag')
    parser.add_argument('--work_dir', type=str, default='')

    args = parser.parse_args()
    args.cfg_env_source = args.work_dir + args.cfg_env_source
    args.cfg_env_source_init = args.work_dir + args.cfg_env_source_init
    args.cfg_env_target = args.work_dir + args.cfg_env_target
    args.cfg_rl = args.work_dir + args.cfg_rl
    args.cfg_simopt = args.work_dir + args.cfg_simopt
    args.rl_logdir = args.work_dir + args.rl_logdir
    args.real_data_path = args.work_dir + args.real_data_path
    args.iter_flag = args.work_dir + args.iter_flag
    args.run_rl_flag = args.work_dir + args.run_rl_flag
    args.run_simopt_flag = args.work_dir + args.run_simopt_flag

    cfg_rl = YamlConfig(args.cfg_rl)
    cfg_simopt = YamlConfig(args.cfg_simopt)
    cfg_env_target = YamlConfig(args.cfg_env_target)

    run_rl_flag = os.path.realpath(args.run_rl_flag)
    run_simopt_flag = os.path.realpath(args.run_simopt_flag)
    iter_flag = os.path.realpath(args.iter_flag)

    run(args.num_iter, args.cfg_env_source, args.cfg_env_source_init, cfg_env_target,
        cfg_rl, cfg_simopt, args.rl_logdir, args.real_data_path,
        iter_flag, run_rl_flag, run_simopt_flag)