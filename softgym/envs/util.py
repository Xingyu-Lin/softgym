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
    '''
    
    if relative is None:
        relative = pos - center
    
    quat = Quaternion(axis=axis, angle=angle)
    after_rotate = quat.rotate(relative)
    return after_rotate + center
