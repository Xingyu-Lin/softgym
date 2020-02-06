from softgym.registered_env import *
import numpy as np

env_performance_bound = {'ClothDrop': 0.,
                         'PourWater': 1.,
                         'PassWater': 0.,
                         'ClothFlatten': 2.28,
                         'ClothFold': -0.065,
                         'RopeFlatten': 4.95,
                         # 'PourWaterGoal': 1.,
                         # 'PassWaterGoal': 0.,
                         # 'ClothDropGoal': 0.,
                         # 'ClothManipulate': ClothManipulateEnv,
                         # 'RopeManipulate': RopeManipulateEnv,
                         # 'ClothFoldGoal': ClothFoldGoalConditionedEnv,
                         }


def make_env(env_name):
    env_kwargs = env_arg_dict[env_name]
    env_kwargs['render'] = True
    env_kwargs['headless'] = False
    return SOFTGYM_ENVS[env_name](**env_kwargs)


def get_cloth_flatten_maximum():
    env = make_env('ClothFlatten')
    covered_areas = []
    for i in range(1000):
        env.reset()
        covered_area = env._set_to_flatten()
        covered_areas.append(covered_area)
    return np.mean(np.array(covered_areas))


def get_cloth_fold_maximum():
    performances = []
    env = make_env('ClothFold')
    for i in range(1000):
        env.reset()
        performance = env._set_to_folded()
        performances.append(performance)
    return np.mean(np.array(performances))


def get_rope_flatten_maximum():
    import pyflex
    env = make_env('RopeFlatten')
    env.reset()
    while 1:  # Straighten the rope by your self
        pyflex.step()
        print(env._get_endpoint_distance())


if __name__ == '__main__':
    # print(get_cloth_flatten_maximum())
    # print(get_cloth_fold_maximum())
    print(get_rope_flatten_maximum())
