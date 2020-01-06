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

