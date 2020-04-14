import os, sys, argparse
import json
import numpy as np
import pickle, time

from autolab_core import YamlConfig

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout_flex import RolloutWorker

from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, n_rollouts, policy_save_interval,
          save_policies, **kwargs):
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            for _ in range(n_rollouts):
                episode = rollout_worker.generate_rollouts()
                policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, val)
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, val)
        for key, val in policy.logs():
            logger.record_tabular(key, val)

        logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = evaluator.current_success_rate()
        if success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        print('Time taken for epoch: {}'.format(time.time() - epoch_start_time))


def launch(cfg_env, cfg_train, logdir, save_policies=True, override_params={}):

    # Configure logging
    if logdir or logger.get_dir() is None:
        logger.configure(dir=logdir)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    set_global_seeds(cfg_train['seed'])

    set_flex_bin_path(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = FlexVecEnv(cfg_env)
    env.set_seed(cfg_train['seed'])
    config.cached_set_flex_env(cfg_env['scene_name'], env)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = cfg_env['scene_name']
    params['replay_strategy'] = cfg_train['learn']['replay_strategy']
    params['replay_k'] = cfg_train['learn']['replay_k']
    if cfg_env['scene_name'] in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[cfg_env[
            'scene_name']])  # merge env-specific parameters in
    params['n_cycles'] = cfg_train['learn']['n_cycles']
    params['n_batches'] = cfg_train['learn']['n_batches']
    params['n_rollouts'] = cfg_train['learn']['n_rollouts']
    params['activation'] = cfg_train['learn']['activation']
    params['layers'] = cfg_train['learn']['layers']
    params['hidden'] = cfg_train['learn']['hidden_units']
    # params['relative_goals'] = cfg_env['scene']['RelativeTarget']
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params_flex(params)
    config.log_params(params, logger=logger)

    dims = {
        'o': env.num_obs,
        'u': env.num_acts,
        'g': env.num_goal,
        'on': env.num_obs_normalize,
        'gn': env.num_goal_normalize,
    }

    policy = config.configure_ddpg(
        dims=dims,
        params=params,
        clip_return=cfg_train['learn']['clip_return'])

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(env, policy, dims, logger, **rollout_params)
    evaluator = RolloutWorker(env, policy, dims, logger, **eval_params)

    train(
        logdir=logdir,
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        n_epochs=cfg_train['learn']['n_epochs'],
        n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'],
        n_batches=params['n_batches'],
        n_rollouts=params['n_rollouts'],
        policy_save_interval=cfg_train['learn']['policy_save_interval'],
        save_policies=save_policies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_env', type=str, default='cfg/fetch_lr_her.yaml')
    parser.add_argument('--cfg_train', type=str, default='cfg/train/her.yaml')
    parser.add_argument('--logdir', type=str, default='logs/her')
    args = parser.parse_args()

    cfg_env = YamlConfig(args.cfg_env)
    cfg_train = YamlConfig(args.cfg_train)
    logdir = os.path.realpath(args.logdir)

    launch(cfg_env, cfg_train, logdir)
