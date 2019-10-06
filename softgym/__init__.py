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
        id='ClothFoldPointControl-v0',
        entry_point='softgym.envs.cloth_fold:ClothFoldPointControlEnv',
        # tags={
        #     'git-commit-hash': '2d95c75',
        #     'author': 'murtaza'
        # },
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'key_point',
        },
    )
