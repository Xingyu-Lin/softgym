import numpy as np
from pyquaternion import Quaternion


def rotate_rigid_object(center, axis, angle, pos = None, relative = None):
    '''
    rotate a rigid object (e.g. shape in flex).

    pos: np.ndarray 3x1, [x, y, z] coordinate of the object.
    relative: relative coordinate of the object to center.
    center: rotation center.
    axis: rotation axis.
    angle: rotation angle in radius.
    TODO: add rotaion of coordinates
    '''

    if relative is None:
        relative = pos - center
    
    quat = Quaternion(axis=axis, angle=angle)
    after_rotate = quat.rotate(relative)
    return after_rotate + center


def quatFromAxisAngle(axis, angle):
    '''
    given a rotation axis and angle, return a quatirian that represents such roatation.
    '''
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat