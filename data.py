"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
import os.path
import random
import numbers
from skimage.util import random_noise
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import torch.utils.data as data

from PIL import Image
import os
import os.path


class RandomCrop(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img1, spectral):
        img_h, img_w, c = img1.shape
        crop_h = np.random.randint(0, img_h - self.w)
        crop_w = np.random.randint(0, img_w - self.h)
        img1 = img1[crop_h:crop_h + self.h, crop_w:crop_w + self.w, :]
        spectral = spectral[:, crop_h:crop_h + self.h, crop_w:crop_w + self.w]
        return img1, spectral


class RandomCrop_3(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img1, img2, img3):
        img_h, img_w, c = img1.shape
        crop_h = np.random.randint(0, img_h - self.w)
        crop_w = np.random.randint(0, img_w - self.h)
        img1 = img1[crop_h:crop_h + self.h, crop_w:crop_w + self.w, :]
        img2 = img2[crop_h:crop_h + self.h, crop_w:crop_w + self.w, :]
        img3 = img3[crop_h:crop_h + self.h, crop_w:crop_w + self.w, :]
        return img1, img2, img3


def default_loader(path):
    return Image.open(path).convert('RGB')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
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


class ImageFolder2D(data.Dataset):
    def __init__(self, root, noise_root, transform, return_paths, train, height, width, crop,
                 loader=default_loader):
        self.noise_root = noise_root
        self.noises = [os.path.join(self.noise_root, x) for x in os.listdir(self.noise_root) if '.npy' in x]
        self.train, self.height, self.width, self.crop = train, height, width, crop
        self.names = [os.path.join(root, x) for x in os.listdir(root)]
        self.spectrals = {}
        # for img_name in tqdm(self.names[:50]):
        #     self.spectrals[img_name] = np.load(os.path.join(img_name, 'spectral.npy'))
        self.root = root
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.random_crop = RandomCrop(height, width)

    def __getitem__(self, index):
        img_name = self.names[index]
        noise_idx = random.randint(0, len(self.noises) - 1)
        noise_dir = self.noises[noise_idx]

        # if img_name in self.spectrals:
        #     spectral = self.spectrals[img_name]
        # else:
        spectral = np.load(os.path.join(img_name, 'spectral.npy')).transpose(2, 0, 1)

        l, W, H = spectral.shape
        gt = Image.open(os.path.join(img_name, 'gt_15S5C.png'))
        # gt = Image.open(os.path.join(img_name, 'gt_EO2113C.png'))
        gt = np.array(gt)
        
        if self.crop:
            gt, spectral = self.random_crop(gt, spectral)
        
        noise = np.load(noise_dir)

        w, h, c = gt.shape
        # print(gt.shape)
        noise_h, noise_w, c = noise.shape
        crop_h = np.random.randint(0, noise_h - h)
        crop_w = np.random.randint(0, noise_w - w)
        noise = noise[crop_h:crop_h + h, crop_w:crop_w + w, :]

        gt = torch.tensor(gt).permute(2, 0, 1) / 255.
        spectral = torch.tensor(spectral)[2:, :, :].float()
        noise = torch.tensor(noise).float().permute(2, 0, 1) / 65535

        if not self.train:
            return spectral, gt, noise, img_name
        else:
            return spectral, gt, noise

    def __len__(self):
        return len(self.names)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class data_prefetcher():
    def __init__(self, loader):
        self.stream = torch.cuda.Stream()
        self.loader = sample_data(loader)
        self.preload()
 
    def preload(self):
        try:
            self.spectral, self.gt, self.noise = next(self.loader)
        except StopIteration:
            self.spectral = None
            self.gt = None
            self.noise = None
            return
        with torch.cuda.stream(self.stream):
            self.spectral = self.spectral.cuda(non_blocking=True).float()
            self.gt = self.gt.cuda(non_blocking=True).float()
            self.noise = self.noise.cuda(non_blocking=True).float()
 
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        spectral = self.spectral
        gt = self.gt
        noise = self.noise
        self.preload()
        return spectral, gt, noise



class ImageFolderEn(data.Dataset):
    def __init__(self, root, transform, return_paths, train, height, width, crop,
                 loader=default_loader):
        # print(name)
        # assert False
        self.train, self.height, self.width, self.crop = train, height, width, crop
        self.names = [os.path.join(root, x) for x in os.listdir(root)]
        self.root = root
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.random_crop = RandomCrop_3(height, width)

    def __getitem__(self, index):
        scene_name = self.names[index]

        gt = Image.open(os.path.join(scene_name, 'gt.png'))
        gt = np.array(gt)

        vis = Image.open(os.path.join(scene_name, 'vis.png'))
        vis = np.array(vis)

        nir = Image.open(os.path.join(scene_name, 'nir.png'))
        nir = np.array(nir)
        
        if self.crop:
            gt, vis, nir = self.random_crop(gt, vis, nir)

        gt = torch.tensor(gt).permute(2, 0, 1) / 255.
        vis = torch.tensor(vis).permute(2, 0, 1) / 255.
        nir = torch.tensor(nir).permute(2, 0, 1) / 255.

        if not self.train:
            return gt, vis, nir, scene_name
        else:
            return gt, vis, nir

    def __len__(self):
        return len(self.names)
