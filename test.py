import argparse
import os
import torch
import numpy as np
import random
from PIL import Image
import torch.backends.cudnn as cudnn
from torch import nn
from unet import G as Generator
from util import get_config


def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    scenes = [os.path.join(dir, x) for x in os.listdir(dir)]
    return scenes


def get_data(scene_name):

    gt = Image.open(os.path.join(scene_name, 'gt.png'))
    gt = np.array(gt)

    vis = Image.open(os.path.join(scene_name, 'vis.png'))
    vis = np.array(vis)

    nir = Image.open(os.path.join(scene_name, 'nir.png'))
    nir = np.array(nir)

    gt = torch.tensor(gt).permute(2, 0, 1) / 255.
    vis = torch.tensor(vis).permute(2, 0, 1) / 255.
    nir = torch.tensor(nir).permute(2, 0, 1) / 255.
    
    gt, vis, nir = gt.unsqueeze(0), vis.unsqueeze(0), nir.unsqueeze(0)

    return gt, vis, nir, scene_name


def test(opts, config, device, scenelist):
    for i, (scene_root) in enumerate(scenelist):
        print(f'{i}/{len(scenelist)}')

        gt, vis, nir, scene_name = get_data(scene_root)
        gt, vis, nir = gt.to(device), vis.to(device), nir.to(device)
        
        # print(nir_val.shape, isp_val.shape)
        with torch.no_grad():
            output = g(vis, nir)
        
        b_vis = vis[0]
        b_nir = nir[0]
        b_output = output[0]
        b_gt = gt[0]
        b_scene_root = scene_root
        # print(b_scene_root)

        b_vis = b_vis.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).detach().to('cpu', torch.uint8).numpy()
        b_nir = b_nir.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).detach().to('cpu', torch.uint8).numpy()
        b_output = b_output.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).detach().to('cpu', torch.uint8).numpy()
        b_gt = b_gt.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).detach().to('cpu', torch.uint8).numpy()
        b_out = np.hstack([b_vis, b_nir, b_output, b_gt])
        b_out = Image.fromarray(b_out)

        save_to = os.path.join(opts.save_to, os.path.basename(b_scene_root) + '.png')
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        # print(save_to)
        # assert False
        b_out.save(save_to)
        # assert False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--save_to', type=str, required=True, help="outputs path")
    parser.add_argument("--resume", required=True)
    opts = parser.parse_args()

    cudnn.benchmark = True
    setup_seed(521)

    # Load experiment setting
    config = get_config(opts.config)
    device = "cuda"

    g = Generator().to(device).eval()

    config['iterations'] = 0
    ckpt = opts.resume
    state_dict = torch.load(ckpt)
    config['iterations'] = int(state_dict['iterations'])
    g.load_state_dict(state_dict['g'])
    print('Resume from iteration %d' % config['iterations'])

    scenelist = make_dataset(config['testroot'])

    test(opts, config, device, scenelist)
