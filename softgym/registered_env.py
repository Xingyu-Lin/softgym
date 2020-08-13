from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.envs.pour_water_amount import PourWaterAmountPosControlEnv
from softgym.envs.pass_water import PassWater1DEnv
from softgym.envs.pass_water_torus import PassWater1DTorusEnv
from softgym.envs.rope_flatten_new import RopeFlattenNewEnv
from softgym.envs.rope_configuration import RopeConfigurationEnv
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.cloth_drop import ClothDropEnv
from softgym.envs.cloth_fold_crumpled import ClothFoldCrumpledEnv
from softgym.envs.cloth_fold_drop import ClothFoldDropEnv
from softgym.envs.rigid_cloth_fold import RigidClothFoldEnv

# from softgym.multitask_envs_arxived.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
# from softgym.multitask_envs_arxived.pass_water_multitask import PassWater1DGoalConditionedEnv
# from softgym.multitask_envs_arxived.cloth_manipulate import ClothManipulateEnv
# from softgym.multitask_envs_arxived.rope_manipulate import RopeManipulateEnv
# from softgym.multitask_envs_arxived.cloth_drop_multitask import ClothDropGoalConditionedEnv
# from softgym.multitask_envs_arxived.cloth_fold_multitask import ClothFoldGoalConditionedEnv
from collections import OrderedDict

env_arg_dict = {
    'PourWater': {'observation_mode': 'cam_rgb',
                  'action_mode': 'direct',
                  'render_mode': 'fluid',
                  'deterministic': False,
                  'render': True,
                  'action_repeat': 8,
                  'headless': True,
                  'num_variations': 1000,
                  'horizon': 100,
                  'use_cached_states': True,
                  'camera_name': 'default_camera'},
    'PourWaterAmount': {'observation_mode': 'cam_rgb',
                        'action_mode': 'direct',
                        'render_mode': 'fluid',
                        'action_repeat': 8,
                        'deterministic': False,
                        'render': True,
                        'headless': True,
                        'num_variations': 1000,
                        'use_cached_states': True,
                        'horizon': 100,
                        'camera_name': 'default_camera'},

    'RopeClothStraighten': {
        'observation_mode': 'cam_rgb',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 75,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1000,
        'use_cached_states': True,
        'deterministic': False
    },

    'RopeFlattenNew': {
        'observation_mode': 'cam_rgb',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 75,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1000,
        'use_cached_states': True,
        'deterministic': False
    },
    'RopeConfiguration': {'observation_mode': 'cam_rgb',
                          'action_mode': 'picker',
                          'num_picker': 2,
                          'render': True,
                          'headless': True,
                          'horizon': 100,  # this task is harder than just straigtening rope, therefore has larger horizon.
                          'action_repeat': 8,
                          'render_mode': 'cloth',
                          'num_variations': 1000,
                          'use_cached_states': True,
                          'deterministic': False},

    'RigidClothFold': {'observation_mode': 'cam_rgb',
                       'action_mode': 'picker',
                       'num_picker': 2,
                       'render': True,
                       'headless': True,
                       'horizon': 100,
                       'action_repeat': 8,
                       'num_pieces': 2,
                       'num_variations': 1000,
                       'use_cached_states': True,
                       'deterministic': False},
    'RigidClothDrop': dict(observation_mode='cam_rgb',
                           action_mode='picker',
                           num_picker=2,
                           num_pieces=1,
                           render=True,
                           headless=True,
                           horizon=30,
                           action_repeat=16,
                           num_variations=1000,
                           use_cached_states=True,
                           deterministic=False),
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
                     'deterministic': False},
    'ClothFlattenPPP': {'observation_mode': 'cam_rgb',
                        'action_mode': 'pickerpickplace',
                        'num_picker': 2,
                        'render': True,
                        'headless': True,
                        'horizon': 20,
                        'action_repeat': 1,
                        'render_mode': 'cloth',
                        'num_variations': 1000,
                        'use_cached_states': True,
                        'deterministic': False},
    'ClothFoldPPP': {'observation_mode': 'cam_rgb',
                     'action_mode': 'pickerpickplace',
                     'num_picker': 2,
                     'render': True,
                     'headless': True,
                     'horizon': 20,
                     'action_repeat': 1,
                     'render_mode': 'cloth',
                     'num_variations': 1000,
                     'use_cached_states': True,
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
                  'deterministic': False},
    'ClothFoldCrumpled': {'observation_mode': 'cam_rgb',
                          'action_mode': 'picker',
                          'num_picker': 2,
                          'render': True,
                          'headless': True,
                          'horizon': 100,
                          'action_repeat': 8,
                          'render_mode': 'cloth',
                          'num_variations': 1000,
                          'use_cached_states': True,
                          'deterministic': False},
    'ClothFoldDrop': {'observation_mode': 'cam_rgb',
                      'action_mode': 'picker',
                      'num_picker': 2,
                      'render': True,
                      'headless': True,
                      'horizon': 100,
                      'action_repeat': 8,
                      'render_mode': 'cloth',
                      'num_variations': 1000,
                      'use_cached_states': True,
                      'deterministic': False},
    'ClothDrop': dict(observation_mode='cam_rgb',
                      action_mode='picker',
                      num_picker=2,
                      render=True,
                      headless=True,
                      horizon=30,
                      action_repeat=16,
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
                      deterministic=False,
                      num_variations=1000),
    'PassWaterTorus': dict(observation_mode='cam_rgb',
                           action_mode='direct',
                           render=True,
                           headless=True,
                           horizon=75,
                           action_repeat=8,
                           render_mode='torus',
                           deterministic=False,
                           num_variations=1000),
    'TransportTorus': dict(observation_mode='cam_rgb',
                           action_mode='direct',
                           render=True,
                           headless=True,
                           horizon=75,
                           action_repeat=8,
                           render_mode='torus',
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
        'camera_name': 'default_camera'
    },
    "ClothManipulate": dict(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=True,
        horizon=100,
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
        horizon=100,
        action_repeat=8,
        render_mode='cloth',
        num_variations=1000,
        deterministic=False,
    )
}

SOFTGYM_ENVS = OrderedDict({
    'ClothDrop': ClothDropEnv,
    'PourWater': PourWaterPosControlEnv,
    'PourWaterAmount': PourWaterAmountPosControlEnv,
    'PassWater': PassWater1DEnv,
    'PassWaterTorus': PassWater1DTorusEnv,
    'ClothFlatten': ClothFlattenEnv,
    'ClothFold': ClothFoldEnv,
    'ClothFoldCrumpled': ClothFoldCrumpledEnv,
    'ClothFoldDrop': ClothFoldDropEnv,
    'RigidClothFold': RigidClothFoldEnv,
    'RopeFlattenNew': RopeFlattenNewEnv,
    'RopeConfiguration': RopeConfigurationEnv,
    # 'PourWaterGoal': PourWaterPosControlGoalConditionedEnv,
    # 'PassWaterGoal': PassWater1DGoalConditionedEnv,
    # 'ClothDropGoal': ClothDropGoalConditionedEnv,
    # 'ClothManipulate': ClothManipulateEnv,
    # 'RopeManipulate': RopeManipulateEnv,
    # 'ClothFoldGoal': ClothFoldGoalConditionedEnv,
    'ClothFlattenPPP': ClothFlattenEnv,
    'ClothFoldPPP': ClothFoldEnv,
})
