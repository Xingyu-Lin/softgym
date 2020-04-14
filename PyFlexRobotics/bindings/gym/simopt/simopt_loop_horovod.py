#!/usr/bin/env python
import argparse, logging, os, time, pickle, copy
from autolab_core import YamlConfig

# import horovod.tensorflow as hvd
import horovod.keras as hvd
import gym

import tensorflow as tf
import numpy as np
import ruamel.yaml as yaml
import numpy as np
import ruamel.yaml as yaml
import scipy as sp

from collections import OrderedDict
from reps import Reps
from baselines import logger
from baselines.common import set_global_seeds, boolean_flag, tf_util as U
from baselines.ppo1.mlp_hvd_policy import MlpPolicy
from baselines.ppo1.clean_ppo1.PPO1_horovod import PPO1 as PPO1_distr_rollout
from baselines.ppo1.clean_ppo1.PPO1_distributed_training_horovod import PPO1 as PPO1_distr_train

from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env


def traj_segment_generator(pi, env, horizon, stochastic, actions_real, real_data):
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
        sub_t = t % horizon

        if actions_real:
            ac = np.tile(real_data['ac'][0,sub_t % 453], (ob.shape[0], 1))
            vpred = 0
        else:
            ac, vpred = pi.act_parallel(stochastic, ob)

        if t > 0 and sub_t == 0:
            return {"ob": obs.copy(), "rew": rews.copy(), "vpred": vpreds.copy(), "new": news.copy(),
                   "ac": acs.copy(), "prevac": prevacs.copy(), "nextvpred": (vpred * (1 - new)).copy(),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}

        obs[:, sub_t] = ob
        vpreds[:, sub_t] = vpred
        news[:, sub_t] = new
        acs[:, sub_t] = ac
        prevacs[:, sub_t] = prevac

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


def compute_cost(real_data, sim_data, cost_weights,
                 l1_weight, l2_weight, cost_gauss_std, cost_gauss_truncate):

    l1_cost = np.zeros(sim_data['ob'].shape[0])
    l2_cost = np.zeros(sim_data['ob'].shape[0])

    real_smooth = sp.ndimage.filters.gaussian_filter1d(real_data['ob'][0],
                        cost_gauss_std, axis=0, truncate=cost_gauss_truncate)

    for ai in range(sim_data['ob'].shape[0]):
        sim_smooth = sp.ndimage.filters.gaussian_filter1d(sim_data['ob'][ai],
                            cost_gauss_std, axis=0, truncate=cost_gauss_truncate)

        for t in range(sim_data['ob'].shape[1]):
            diff = cost_weights * (sim_smooth[t] - real_smooth[t])
            l1_cost[ai] += np.sum(np.abs(diff))
            l2_cost[ai] += np.linalg.norm(diff)

    cost = l1_weight * l1_cost + l2_weight * l2_cost

    logger.log("Compute SimOpt cost. Real shape " + str(real_data['ob'].shape)
           + " Sim shape " + str(sim_data['ob'].shape) + " Cost Min " + str(np.min(cost))
           + " Cost Max " + str(np.max(cost)))
    return cost


def yaml_ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def optimize(cfg_simopt, cfg_rl, cfg_env, rl_logdir,
             real_data_path, result_cfg_env_path, optim_algo):
    seed = cfg_rl['seed'] + hvd.rank()
    set_global_seeds(seed)
    cfg_env['gym']['seed'] = seed
    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))

    def policy_fn(name, ob_space, ac_space):
        try:
            policy_class = eval(cfg_rl['policy_type'])
        except:
            logging.error('Policy type {} not found!'.format(cfg_rl['policy_type']))
            exit()
        return policy_class(name=name, ob_space=ob_space, ac_space=ac_space, **cfg_rl['policy'])

    policy_path = os.path.join(rl_logdir, "{}-{}".format(cfg_rl['learn']['agent_name'],
                                                         cfg_rl['learn']['max_iters']))
    # Sample sim parameters
    current_mean = np.asarray(cfg_env['scene']['SimParamsMean'])
    current_min = np.asarray(cfg_env['scene']['SimParamsMin'])
    current_max = np.asarray(cfg_env['scene']['SimParamsMax'])

    if 'SimParamsCov' in cfg_env['scene']:
        current_cov = np.asarray(cfg_env['scene']['SimParamsCov'])
    else:
        current_cov = np.diag(cfg_env['scene']['SimParamsCovInit'])

    real_data_f = open(real_data_path, 'rb')
    real_data = pickle.load(real_data_f)
    real_data_f.close()

    sim_params_stochastic_orig = cfg_env['scene']['SimParamsStochastic']
    cfg_env['scene']['SimParamsStochastic'] = False
    env = make_flex_vec_env(cfg_env)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    with tf.Session(config=config).as_default():
        pi = load_policy(env, policy_fn, policy_path)
        env.close()

        num_agents = cfg_env['scene']['NumAgents']
        cfg_env['scene']['SimParamsStochastic'] = True
        cfg_env['scene']['SimParamsPresampled'] = True
        num_samples = cfg_simopt['learn']['num_samples']
        cost_weights = cfg_simopt['learn']['cost_weights']
        max_cost = cfg_simopt['learn']['max_cost']
        actions_real = cfg_simopt['learn']['actions_real']
        l1_weight = cfg_simopt['learn']['l1_weight']
        l2_weight = cfg_simopt['learn']['l2_weight']
        cost_gauss_std = cfg_simopt['learn']['cost_gauss_std']
        cost_gauss_truncate = cfg_simopt['learn']['cost_gauss_truncate']

        for optim_iter in range(0, cfg_simopt['learn']['num_iter']):
            samples = np.random.multivariate_normal(current_mean, current_cov, num_samples)
            samples = np.maximum(samples, current_min)
            samples = np.minimum(samples, current_max)

            costs = np.zeros(num_samples)
            for i in range(0, num_samples, num_agents):
                cfg_env['scene']['SimParamsSamples'] = samples[i:i+num_agents].tolist()
                env = make_flex_vec_env(cfg_env)
                seg = traj_segment_generator(pi, env, cfg_simopt['learn']['timesteps_per_batch'],
                                             cfg_simopt['learn']['policy_stochastic'],
                                             actions_real, real_data)
                costs[i:i+num_agents] = compute_cost(real_data, seg, cost_weights, l1_weight,
                                                     l2_weight, cost_gauss_std, cost_gauss_truncate)
                env.close()
                print("Rank " + str(hvd.rank()) + ". Iteration " +
                           str(optim_iter) + " finish samples " + str(i + num_agents))

            samples = hvd.allgather(samples)
            costs = hvd.allgather(costs)

            if hvd.rank() == 0:

                # Replace NaNs with max cost * 10.0
                costs_nanmax = np.nanmax(costs, axis=0)
                costs[np.where(np.isnan(costs))] = costs_nanmax * 10.0

                cost_mask = np.argwhere(costs > max_cost).flatten()
                costs = np.delete(costs, cost_mask, axis=0)
                samples = np.delete(samples, cost_mask, axis=0)

                print("SIZE SAMPLES ", str(samples.shape), " COST ", str(costs.shape))
                assert(len(samples) == len(costs))

                current_mean, current_cov = optim_algo.learn(samples, costs, current_mean, current_cov)
                logger.record_tabular("simopt_mean", float(np.mean(costs)))
                logger.record_tabular("simopt_std", float(np.std(costs)))
                logger.record_tabular("simopt_optim_iter", optim_iter)
                logger.record_tabular("simopt_num_samples", len(samples))
                logger.dump_tabular()

            current_mean = hvd.broadcast(current_mean, 0)
            current_cov = hvd.broadcast(current_cov, 0)
        U._PLACEHOLDER_CACHE = {}
        tf.reset_default_graph()

    if hvd.rank() == 0:
        # Save updated parameters
        # cfg_env['gym']['renderBackend'] = 0
        cfg_env['scene']['SimParamsStochastic'] = sim_params_stochastic_orig
        cfg_env['scene']['SimParamsPresampled'] = False
        cfg_env['scene']['SimParamsMean'] = current_mean.tolist()
        cfg_env['scene']['SimParamsCov'] = current_cov.tolist()
        cfg_env['exp']['SimOptCostMean'] = np.asscalar(np.mean(costs))
        cfg_env['exp']['SimOptCostStd'] = np.asscalar(np.std(costs))

        result_yaml = open(result_cfg_env_path, 'w+')
        yaml_ordered_dump(cfg_env.config, stream=result_yaml, Dumper=yaml.SafeDumper)
        result_yaml.close()

    import gc
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_simopt', type=str, default='/sim2real/cfg/simopt.yaml')
    parser.add_argument('--cfg_rl', type=str, default='/sim2real/cfg/ppo1.yaml')
    parser.add_argument('--cfg_env', type=str, default='/sim2real/log/franka_cabinet')

    parser.add_argument('--simopt_logdir', type=str, default='/sim2real/log/simopt')
    parser.add_argument('--rl_logdir', type=str, default='/sim2real/log/ppo1')
    parser.add_argument('--real_data_path', type=str, default='/sim2real/log/real')

    parser.add_argument('--iter_flag', type=str, default='/sim2real/log/iter.flag')
    parser.add_argument('--run_simopt_flag', type=str, default='/sim2real/log/run_simopt.flag')

    parser.add_argument('--work_dir', type=str, default='')

    args = parser.parse_args()
    args.cfg_simopt = args.work_dir + args.cfg_simopt
    args.cfg_rl = args.work_dir + args.cfg_rl
    args.cfg_env = args.work_dir + args.cfg_env
    args.simopt_logdir = args.work_dir + args.simopt_logdir
    args.rl_logdir = args.work_dir + args.rl_logdir
    args.real_data_path = args.work_dir + args.real_data_path
    args.iter_flag = args.work_dir + args.iter_flag
    args.run_simopt_flag = args.work_dir + args.run_simopt_flag

    cwd_orig = os.getcwd()
    simopt_logdir = os.path.realpath(args.simopt_logdir)
    run_simopt_flag = os.path.realpath(args.run_simopt_flag)
    iter_flag = os.path.realpath(args.iter_flag)

    try:
        os.remove(run_simopt_flag)
    except OSError:
        pass

    hvd.init()
    if hvd.rank() == 0:
        logging.getLogger().setLevel(logging.INFO)
        os.makedirs(simopt_logdir, exist_ok=True)
        logger.configure(simopt_logdir)
    while True:
        print("Rank " + str(hvd.rank()) + ". Waiting for SimOpt requests...")
        if os.path.isfile(run_simopt_flag):
            with open(iter_flag, 'r') as f:
                cur_iter = f.read().replace('\n', '')
            print("Rank " + str(hvd.rank()) +
                  ". SimOpt request received. Iteration " + cur_iter)

            cfg_simopt = YamlConfig(args.cfg_simopt)
            cfg_rl = YamlConfig(args.cfg_rl)
            reps_optim = Reps(kl_threshold=cfg_simopt['learn']['kl_threshold'],
                              covariance_damping=cfg_simopt['learn']['covariance_damping'],
                              min_temperature=cfg_simopt['learn']['min_temperature'])

            cfg_env_path = os.path.realpath(args.cfg_env + "_" + cur_iter + ".yaml")
            cfg_env = YamlConfig(cfg_env_path)

            rl_logdir = os.path.realpath(args.rl_logdir + "_" + cur_iter)
            result_cfg_env_path = os.path.realpath(
                args.cfg_env + "_" + str(int(cur_iter) + 1) + ".yaml")

            real_data_path_iter = os.path.realpath(args.real_data_path +
                                                   "_" + cur_iter + ".pkl")

            optimize(cfg_simopt, cfg_rl, cfg_env, rl_logdir,
                     real_data_path_iter, result_cfg_env_path, reps_optim)

            os.chdir(cwd_orig)
            try:
                os.remove(run_simopt_flag)
            except OSError:
                pass

            print("Rank " + str(hvd.rank()) +
                  ". Finished SimOpt. Iteration " + cur_iter)
        time.sleep(2.0)
