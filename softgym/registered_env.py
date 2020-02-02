from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.envs.pass_water import PassWater1DEnv
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.cloth_drop import ClothDropEnv
from softgym.envs.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
from softgym.envs.pass_water_multitask import PassWater1DGoalConditionedEnv
from softgym.envs.cloth_manipulate import ClothManipulateEnv
from softgym.envs.rope_manipulate import RopeManipulateEnv
from softgym.envs.cloth_drop_multitask import ClothDropGoalConditionedEnv
from softgym.envs.cloth_fold_multitask import ClothFoldGoalConditionedEnv
from collections import OrderedDict

env_arg_dict = {
    'PourWater': {'observation_mode': 'cam_rgb',
                  'action_mode': 'direct',
                  'render_mode': 'fluid',
                  'deterministic': False,
                  'render': True,
                  'headless': True,
                  'num_variations': 1000,
                  'horizon': 100,
                  'delta_reward': False,
                  'camera_name': 'default_camera'},
    'RopeFlatten': {'observation_mode': 'cam_rgb',
                    'action_mode': 'picker',
                    'num_picker': 2,
                    'render': True,
                    'headless': True,
                    'horizon': 75,
                    'action_repeat': 8,
                    'render_mode': 'cloth',
                    'num_variations': 1000,
                    'use_cached_states': True,
                    'delta_reward': False,
                    'deterministic': False},
    'ClothFlatten': {'observation_mode': 'cam_rgb',
                     'action_mode': 'picker',
                     'num_picker': 2,
                     'render': True,
                     'headless': True,
                     'horizon': 100,
                     'action_repeat': 8,
                     'render_mode': 'cloth',
                     'num_variations': 1000,
                     'use_cached_states': True,
                     'delta_reward': False,
                     'deterministic': False},
    'ClothFold': {'observation_mode': 'cam_rgb',
                  'action_mode': 'picker',
                  'num_picker': 2,
                  'render': True,
                  'headless': True,
                  'horizon': 100,
                  'action_repeat': 8,
                  'render_mode': 'cloth',
                  'num_variations': 1000,
                  'use_cached_states': True,
                  'delta_reward': False,
                  'deterministic': False},
    'ClothDrop': dict(observation_mode='cam_rgb',
                      action_mode='picker',
                      num_picker=2,
                      render=True,
                      headless=True,
                      horizon=15,
                      action_repeat=32,
                      render_mode='cloth',
                      num_variations=1000,
                      use_cached_states=True,
                      deterministic=False),
    'PassWater': dict(observation_mode='cam_rgb',
                      action_mode='direct',
                      render=True,
                      headless=True,
                      horizon=75,
                      action_repeat=8,
                      render_mode='fluid',
                      delta_reward=False,
                      deterministic=False,
                      num_variations=1000),
    'PassWaterGoal': {
        "observation_mode": 'point_cloud',  # will be later wrapped by ImageEnv
        "horizon": 75,
        "action_mode": 'direct',
        "deterministic": False,
        "render_mode": 'fluid',
        "render": True,
        "headless": True,
        "action_repeat": 8,
        "delta_reward": False,
        "num_variations": 1000,
    },
    "PourWaterGoal": {
        'observation_mode': 'point_cloud',
        'action_mode': 'direct',
        'render_mode': 'fluid',
        'deterministic': False,
        'render': True,
        'headless': True,
        'num_variations': 1000,
        'horizon': 100,
        'delta_reward': False,
        'camera_name': 'default_camera'
    },
    "ClothManipulate": dict(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=True,
        horizon=150,
        action_repeat=8,
        render_mode='cloth',
        num_variations=1000,
        deterministic=False
    ),
    "ClothDropGoal": dict(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=True,
        horizon=15,
        action_repeat=32,
        render_mode='cloth',
        num_variations=1000,
        deterministic=False
    ),
    "RopeManipulate": dict(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=True,
        horizon=75,
        action_repeat=8,
        render_mode='rope',
        num_variations=1000,
        deterministic=False
    ),
    "ClothFoldGoal": dict(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=True,
        horizon=150,
        action_repeat=8,
        render_mode='cloth',
        num_variations=1000,
        deterministic=False,
        delta_reward=False,
    )
}

SOFTGYM_ENVS = OrderedDict({
    'ClothDrop': ClothDropEnv,
    'PourWater': PourWaterPosControlEnv,
    'PassWater': PassWater1DEnv,
    'ClothFlatten': ClothFlattenEnv,
    'ClothFold': ClothFoldEnv,
    'RopeFlatten': RopeFlattenEnv,
    'PourWaterGoal': PourWaterPosControlGoalConditionedEnv,
    'PassWaterGoal': PassWater1DGoalConditionedEnv,
    'ClothDropGoal': ClothDropGoalConditionedEnv,
    'ClothManipulate': ClothManipulateEnv,
    'RopeManipulate': RopeManipulateEnv,
    'ClothFoldGoal': ClothFoldGoalConditionedEnv,
})
