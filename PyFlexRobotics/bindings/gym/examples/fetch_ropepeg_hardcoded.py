import os, sys, logging
import numpy as np
from autolab_core import YamlConfig

from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
def unscale(lo, hi, y):
    return 2. * (y - lo) / (hi - lo) - 1.

def unscale_action(target_pose):
    return np.array([
        unscale(-0.5, 0.5, target_pose[0]),
        unscale(0., 1.6, target_pose[1]),
        unscale(0., 1.2, target_pose[2]),
        unscale(-np.pi, np.pi, target_pose[3]),
        unscale(-np.pi, np.pi, target_pose[4]),
        unscale(-np.pi, np.pi, target_pose[5]),
        unscale(0.002, 0.05, target_pose[6])
    ])

def get_action_poses(obs):
    holder_loc = obs[0, 7:10]
    vert_orient = np.array([0, 0, np.deg2rad(-90)])
    approach_pose = np.r_[holder_loc + np.array([0, 0.6, 0]), vert_orient,
                              [0.03]]
    insert_pose = np.r_[holder_loc + np.array([0, 0.4, 0]), vert_orient,
                            [0.03]]

    return approach_pose, insert_pose


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    cfg = YamlConfig('cfg/fetch_ropepeg.yaml')
    cfg['scene']['NumAgents'] = 1

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = FlexVecEnv(cfg)

    t = 0
    obs = env.reset()
    approach_pose, insert_pose = get_action_poses(obs)
    while True:
        if t > 1300:
            action = unscale_action(insert_pose)
        else:
            action = unscale_action(approach_pose)
        actions = np.array([action])
        obs, rews, dones, _ = env.step(actions)
        logging.info('mean reward: {:.3f}'.format(np.mean(rews)))
        sys.stdout.flush()

        t += 1
        if dones[0]:
            t = 0
            obs = env.reset()
            approach_pose, insert_pose = get_action_poses(obs)
