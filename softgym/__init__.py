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

    register(
        id='ClothFoldSphereControl-v0',
        entry_point='softgym.envs.cloth_fold:ClothFoldPointControlEnv',
        # tags={
        #     'git-commit-hash': '2d95c75',
        #     'author': 'murtaza'
        # },
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'sphere',
            'render': True,
            'headless': False
        },
    )

    register(
        id='ClothFoldForceControl-v0',
        entry_point='softgym.envs.cloth_fold:ClothFoldPointControlEnv',
        # tags={
        #     'git-commit-hash': '2d95c75',
        #     'author': 'murtaza'
        # },
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'force',
        },
    )

    register(
        id='ClothFoldStickyControl-v0',
        entry_point='softgym.envs.cloth_fold:ClothFoldPointControlEnv',
        # tags={
        #     'git-commit-hash': '2d95c75',
        #     'author': 'murtaza'
        # },
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'sticky',
        },
    )

    register(
        id='ClothFoldBoxControl-v0',
        entry_point='softgym.envs.cloth_fold:ClothFoldPointControlEnv',
        # tags={
        #     'git-commit-hash': '2d95c75',
        #     'author': 'murtaza'
        # },
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'block',
        },
    )

    register(
        id='ClothFlattenPointControl-v0',
        entry_point='softgym.envs.cloth_flatten:ClothFlattenPointControlEnv',
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'key_point_pos',

        }
    )

    register(
        id='ClothFlattenSphereControl-v0',
        entry_point='softgym.envs.cloth_flatten:ClothFlattenPointControlEnv',
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'sphere',
            'render': False,
            'headless': True
        }
    )

    register(
        id='ClothFlattenSphereControlGoalConditioned-v0',
        entry_point='softgym.envs.cloth_flatten_multitask:ClothFlattenPointControlGoalConditionedEnv',
        kwargs={
            'observation_mode': 'key_point',
            'action_mode': 'sphere',
            'horizon': 100
        }
    )

    register(
        id='PourWaterPosControl-v0',
        entry_point='softgym.envs.pour_water:PourWaterPosControlEnv',
        kwargs={
            'observation_mode': 'cam_img',
            'action_mode': 'direct',
            'render_mode': 'fluid',
            'deterministic': True,
            'render': True,
            'headless': False,
            'horizon': 75
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
