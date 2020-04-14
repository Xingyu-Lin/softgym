import os, argparse, logging

from autolab_core import YamlConfig

from baselines.ars.ars_flex import ARSLearner
from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv


def run_ars(cfg_env, cfg_train, logdir):

    logdir = os.path.realpath(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    set_flex_bin_path(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    cfg_env['gym']['seed'] = cfg_train['seed']
    env = FlexVecEnv(cfg_env)

    policy_params = cfg_train['policy']
    policy_params.update({
        'ob_dim': env.num_obs,
        'ac_dim': env.num_acts,
    })

    ARS = ARSLearner(
        env=env,
        policy_params=policy_params,
        logdir=logdir,
        **cfg_train['learn']
        )

    ARS.train()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_env', type=str, default='cfg/ant.yaml')
    parser.add_argument('--cfg_train', type=str, default='cfg/train/ars.yaml')
    parser.add_argument('--logdir', type=str, default='logs/ars')
    args = parser.parse_args()

    cfg_env = YamlConfig(args.cfg_env)
    cfg_train = YamlConfig(args.cfg_train)

    run_ars(cfg_env, cfg_train, args.logdir)
