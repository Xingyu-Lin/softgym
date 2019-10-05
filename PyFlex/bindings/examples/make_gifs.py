import os
import cv2
import numpy as np
import imageio
import argparse
import scipy.misc


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='')
parser.add_argument('--st_idx', type=int, default=1)
parser.add_argument('--ed_idx', type=int, default=300)
parser.add_argument('--height', type=int, default=240)
parser.add_argument('--width', type=int, default=320)

args = parser.parse_args()

if args.src_dir[5:] == 'FluidFall':
    st_x, ed_x = 60, 210
elif args.src_dir[5:] == 'BoxBath':
    st_x, ed_x = 65, 215
elif args.src_dir[5:] == 'FluidShake':
    st_x, ed_x = 60, 170
elif args.src_dir[5:] == 'RiceGrip':
    st_x, ed_x = 90, 230
elif args.src_dir[5:] == 'RigidFall':
    st_x, ed_x = 65, 210
elif args.src_dir[5:] == 'FluidIceShake':
    st_x, ed_x = 30, 170

images = []
for i in range(args.st_idx, args.ed_idx):

    filename = os.path.join(args.src_dir, 'render_%d.tga' % i)
    print(filename)
    img = scipy.misc.imread(filename)
    img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)

    # images.append(img[st_x:ed_x])
    images.append(img[:])

imageio.mimsave(args.src_dir + '.gif', images, duration=1./60.)

