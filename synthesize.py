import os
import cv2
import h5py
import numpy as np
import shutil
import scipy.io as io
from tqdm import tqdm


IMG_EXTENSIONS = [
    '.npy',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


lst = make_dataset('/tempssd/VCSD/dataset')

# the camera is in BGR, but since using cv2 to save, do not need [::-1] to trun BGR to RGB
camera = np.load('GS3-U3-15S5C_420_890.npy')
daylight = np.load('white_led_61.npy') # (61)

daylight = daylight[2:-11]
camera = camera[:, :, None, None]
daylight = daylight[None, :, None, None]

for x in tqdm(lst):
    image_responce = np.load(x)[2:, :, :]
    image_responce = image_responce[None, :]
    rgb_img = np.sum(daylight * camera * image_responce, axis=1)
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    rgb_img = rgb_img / np.max(rgb_img)
    # rgb_img *= 3
    rgb_img[rgb_img > 1] = 1
    rgb_img[rgb_img < 0] = 0
    rgb_img *= 255
    # print(rgb_img.shape)
    # assert False
    save_to = x.replace('spectral.npy', 'gt_15S5C.png')
    # print(save_to)
    # assert False
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    cv2.imwrite(save_to, rgb_img)
    # assert False