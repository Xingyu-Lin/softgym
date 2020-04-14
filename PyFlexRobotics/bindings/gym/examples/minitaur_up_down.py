import os, sys, random
from time import sleep, time
import numpy as np

from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv

from autolab_core import YamlConfig

def velocity_control_action(frame, agent, invfrequency=20, velocity=4.):
    action = np.zeros(8)
    action[0] = 2.25 * velocity * (-1) ** (frame // invfrequency)
    action[1] = velocity  * (-1) ** (frame // invfrequency)
    action[4] = -velocity * (-1) ** (frame // invfrequency)
    action[5] = -2.25 * velocity * (-1) ** (frame // invfrequency)

    action[2] = 1.3 * velocity * (-1) ** (frame // invfrequency)
    action[3] = 1.35 * velocity * (-1) ** (frame // invfrequency)
    action[6] = -1.35 * velocity * (-1) ** (frame // invfrequency)
    action[7] = -1.3 * velocity * (-1) ** (frame // invfrequency)

    return action

def position_control_action(action_scale, frequency, frame, phase_shift, back_phase_shift):
    action = np.zeros(8)
    action[0] = action_scale * np.sin(frequency * frames - phase_shift)
    action[1] = action_scale * np.sin(frequency * frames)
    action[4] = -action_scale * np.sin(frequency * frames)
    action[5] = -action_scale * np.sin(frequency * frames - phase_shift)

    action[2] = action_scale * np.sin(frequency * frames - phase_shift - back_phase_shift)
    action[3] = action_scale * np.sin(frequency * frames - back_phase_shift)
    action[6] = -action_scale * np.sin(frequency * frames - back_phase_shift)
    action[7] = -action_scale * np.sin(frequency * frames - phase_shift - back_phase_shift)
    return action

if __name__ == '__main__':
    cfg = YamlConfig('cfg/minitaur.yaml')

    set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
    env = FlexVecEnv(cfg)

    use_velocity_control = True

    # velocity control params
    invfrequency = 40
    velocity = 2.

    # position control params
    phase_shift = -0.33
    back_phase_shift = 0.05
    frequency = 0.1
    action_scale = 1.0

    # Simulation loop
    env.reset()
    frames = -20
    while True:
        actions = []
        for agent in range(env.num_envs):
            if use_velocity_control:
                fr = frames
                if frames < 0:
                    fr = 0
                actions.append(velocity_control_action(fr, agent, invfrequency=invfrequency, velocity=velocity))
            else:
                actions.append(position_control_action(action_scale, frequency, frames, phase_shift, back_phase_shift))

        actions = np.array(actions)
        env.step(actions)

        frames += 1
        if frames > 500:
            env.reset()
            frames -= 515
        else:
            env.reset()

    env.close()
