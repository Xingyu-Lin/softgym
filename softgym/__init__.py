import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_flex_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering all softgym environments")

    register(
        id='PourWaterPosControl-v0',
        entry_point='softgym.envs.pour_water:PourWaterPosControlEnv',
        kwargs={
            'observation_mode': 'cam_rgb',
            'action_mode': 'direct',
            'render_mode': 'fluid',
            'deterministic': True,
            'render': True,
            'headless': False,
            'horizon': 75,
        }
    )

    register(
        id='PourWaterPosControlGoalConditioned-v0',
        entry_point='softgym.envs.pour_water_multitask:PourWaterPosControlGoalConditionedEnv',
        kwargs={
            'observation_mode': 'full_state',
            'action_mode': 'direct',
            'render_mode': 'fluid',
            'deterministic': True,
            'render': True,
            'headless': True,
            'horizon': 75
        }
    )
