import numpy as np
from PIL import Image


def rotation_2d_around_center(pt, center, theta):
    """
    2d rotation on 3d vectors by ignoring y factor
    :param pt:
    :param center:
    :return:
    """
    pt = pt.copy()
    pt = pt - center
    x, y, z = pt
    new_pt = np.array([np.cos(theta) * x - np.sin(theta) * z, y, np.sin(theta) * x + np.cos(theta) * z]) + center
    return new_pt


def extend_along_center(pt, center, add_dist, min_dist, max_dist):
    pt = pt.copy()
    curr_dist = np.linalg.norm(pt - center)
    pt = pt - center
    new_dist = min(max(min_dist, curr_dist + add_dist), max_dist)
    pt = pt * (new_dist / curr_dist)
    pt = pt + center
    return pt


def vectorized_range(start, end):
    """  Return an array of NxD, iterating from the start to the end"""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)[:, None] / N + start[:, None]).astype('int')
    return idxes


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y
