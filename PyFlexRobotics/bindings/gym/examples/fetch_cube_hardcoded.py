import os, sys, logging
from time import time
import numpy as np
import pdb
from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv

from autolab_core import YamlConfig

def unscale(lo, hi, y):
    return 2. * (y - lo) / (hi - lo) - 1.

def unscale_action(target_pose):
    return np.swapaxes(np.array([
        unscale(-0.5, 0.5, target_pose[:, 0]),
        unscale(0.4, 0.8, target_pose[:, 1]),
        unscale(0.0, 1.0, target_pose[:, 2]),
        unscale(-np.pi, np.pi, target_pose[:, 3]),
        unscale(-np.pi, np.pi, target_pose[:, 4]),
        unscale(-np.pi, np.pi, target_pose[:, 5]),
        unscale(0.002, 0.05, target_pose[:, 6])
    ]),0,1)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    cfg = YamlConfig('cfg/fetch_cube.yaml')

    numAgents = cfg['scene']['NumAgents'] = 1
    cfg['scene']['NumPerRow'] = np.sqrt(np.floor(numAgents))
    cfg['scene']['SampleInitStates'] = False
    cfg['scene']['InitialGrasp'] = False
    cfg['scene']['RelativeTarget'] = False
    cfg['scene']['DoDeltaPlanarControl'] = False
    cfg['scene']['DoGripperControl'] = True

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = FlexVecEnv(cfg)

    obs = env.reset()

    open_width = 0.05
    grasp_width = 0.023

    rest_pose = np.zeros((numAgents, 7))
    prep_pose = np.zeros((numAgents, 7))
    reach_pose = np.zeros((numAgents, 7))
    grasp_pose = np.zeros((numAgents, 7))
    ball_pose = np.zeros((numAgents, 7))
    gripper_loc = np.zeros((numAgents, 3))
    cube_loc = obs[:, 10:13]
    ball_loc = obs[:, 7:10]
    print(cube_loc)
    print(ball_loc)
    for i in range(numAgents):
        rest_pose[i] = np.r_[cube_loc[i] + np.array([-0.3, 0.3, 0.0]), [0., 0., -np.pi/2, open_width]]
        prep_pose[i] = np.r_[cube_loc[i] + np.array([0.0, 0.2, 0.0]), [0., 0., -np.pi/2, open_width]]
        reach_pose[i] = np.r_[cube_loc[i] + np.array([0.0, 0.11, 0.0]), [0., 0., -np.pi/2, open_width]]
        grasp_pose[i] = np.r_[cube_loc[i] + np.array([0.0, 0.11, 0.0]), [0., 0., -np.pi/2, grasp_width]]
        ball_pose[i] = np.r_[ball_loc[i] + np.array([0.0, 0.12, 0.0]), [0.0, 0.0, -np.pi/2, grasp_width]]

    state = ['start' for i in range(numAgents)]
    target_pose = rest_pose

    s = time()
    while True:
        sys.stdout.flush()
        act = unscale_action(target_pose)

        obs, rew, done, _ = env.step(np.array([act]))

        gripper_loc[:] = obs[:, :3]
        # logging.info('reward: {:.3f}'.format(rew[0]))

        if time() - s < 1:
            continue

        for j in range(numAgents):
            
            if state[j] == 'start':
                if gripper_loc[j, 0] < rest_pose[j, 0]:
                    print('switching to prep: {}'.format(j))
                    state[j] = 'prep'
                    target_pose[j] = prep_pose[j]

            elif state[j] == 'prep':
                # print(np.fabs(gripper_loc[j, 0] - prep_pose[j, 0]))
                if (np.fabs(gripper_loc[j, 0] - prep_pose[j, 0]) < 0.07):
                    print('switching to reach: {}'.format(j))
                    state[j] = 'reach'
                    target_pose[j] = reach_pose[j]

            elif state[j] == 'reach':
                if gripper_loc[j, 1] < 0.46:
                    print('switching to grasp: {}'.format(j))
                    state[j] = 'grasp'
                    target_pose[j] = grasp_pose[j]

            elif state[j] == 'grasp':
                if obs[j, 6] < grasp_width + 0.0005:
                    print('switching to ball: {}'.format(j))
                    state[j] = 'ball'
                    target_pose[j] = ball_pose[j]
