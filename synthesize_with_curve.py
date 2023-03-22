import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import argparse


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--camera', type=str, required=True)
    parser.add_argument('--whitelight', type=str, required=True)
    opts = parser.parse_args()

    lst = make_dataset(opts.root)

    whitelight = np.load(opts.whitelight)[2:-11]
    whitelight = whitelight[None, :, None, None]

    designed_light = torch.load(opts.ckpt_path)['curve'].detach().cpu().numpy()
    camera = np.load(opts.camera)[::-1, :]

    camera = camera[:, :, None, None]
    designed_light = designed_light[None, :, None, None]

    for x in tqdm(lst):
        image_responce = np.load(x)
        image_responce = image_responce[None, 2:]
        
        vis_led_spectral = designed_light[:, :28] # [420nm-690nm]
        vis_camera_response = camera[:, :28] # [420nm-690nm]
        vis_image_responce = image_responce[:, :28] # [420nm-690nm]

        nir_led_spectral = designed_light # [700-890nm]
        nir_camera_response = camera # [700-890nm]
        nir_image_responce = image_responce # [700-890nm]

        gt_img = np.sum(whitelight * camera * image_responce, axis=1)
        gt_img = np.transpose(gt_img, (1, 2, 0))
        gt_img = gt_img / np.max(gt_img) * 255
        gt_img = np.clip(gt_img, 0, 255).astype(np.uint8)

        rgb_img = np.sum(vis_led_spectral * vis_camera_response * vis_image_responce, axis=1)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = rgb_img / np.max(rgb_img)
        
        rgb_img[rgb_img > 1] = 1
        rgb_img[rgb_img < 0] = 0
        rgb_img *= 4095
        poisson= torch.distributions.Poisson(torch.tensor(rgb_img))
        rgb_img = poisson.sample().numpy() / 4095 * 255
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

        nir_img = np.sum(nir_led_spectral * nir_camera_response * nir_image_responce, axis=1)
        nir_img = np.transpose(nir_img, (1, 2, 0))
        nir_img = nir_img / np.max(nir_img)
        
        nir_img[nir_img > 1] = 1
        nir_img[nir_img < 0] = 0
        nir_img *= 4095
        poisson= torch.distributions.Poisson(torch.tensor(nir_img))
        nir_img = poisson.sample().numpy() / 4095 * 255
        nir_img = np.clip(nir_img, 0, 255).astype(np.uint8)

        gt_save_to = x.replace(opts.root, opts.save_root).replace('spectral.npy', 'gt.png')
        vis_save_to = x.replace(opts.root, opts.save_root).replace('spectral.npy', 'vis.png')
        nir_save_to = x.replace(opts.root, opts.save_root).replace('spectral.npy', 'nir.png')

        os.makedirs(os.path.dirname(gt_save_to), exist_ok=True)
        gt_img = Image.fromarray(gt_img)
        gt_img.save(gt_save_to)

        os.makedirs(os.path.dirname(vis_save_to), exist_ok=True)
        rgb_img = Image.fromarray(rgb_img)
        rgb_img.save(vis_save_to)

        os.makedirs(os.path.dirname(nir_save_to), exist_ok=True)
        nir_img = Image.fromarray(nir_img)
        nir_img.save(nir_save_to)