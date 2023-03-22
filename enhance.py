import argparse
import os
import torch
import torch.nn as nn
import shutil
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch import nn
from tqdm import tqdm
import tensorboardX
from unet import G as Generator
from loss import PerceptualLoss
from util import sample_data, get_config, prepare_sub_folder, \
    get_data_loaders_enhance, viz_enhance, make_dataset



def train(
        opts, config, 
        g, opt_g, 
        train_loader, 
        device, checkpoint_directory, image_directory
        ):
	
    train_loader = sample_data(train_loader)

    pbar = range(config['max_iter'])

    pbar = tqdm(pbar, initial=config['iterations'], dynamic_ncols=True, smoothing=0.01)

    percep = PerceptualLoss().to(device)

    for idx in pbar:
        i = idx + config['iterations']

        if i > config['max_iter']:
            print("Done!")

            break
        
        g.train()

        gt, vis, nir = next(train_loader)
        gt, vis, nir = \
            gt.to(device), vis.to(device), nir.to(device)

        opt_g.zero_grad()
        output = g(vis, nir)

        loss_G = percep(output, gt)
        
        loss_G.backward()
        opt_g.step()

        description = f"loss_G: {loss_G:.4f} "
        pbar.set_description((description))
        train_writer.add_scalar('loss_G', loss_G.item(), i)

        if i % config['image_save_iter'] == 0:
            viz_enhance(
                output, gt, vis, nir,
                os.path.join(image_directory, str(i).zfill(7) + '_train_pic.png')
            )

        if i % config['snapshot_save_iter'] == 0:
            save_dict = {
                "g": g.state_dict(),
                "opt_g": opt_g.state_dict(),
                "iterations": i
            }
            torch.save(
                save_dict,
                os.path.join(checkpoint_directory, f"{str(i).zfill(7)}.pt"),
            )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--output_path', type=str, required=True, help="outputs path")
    parser.add_argument("--resume", default=None)
    opts = parser.parse_args()

    cudnn.benchmark = True
    setup_seed(521)

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    checkpoint_directory, image_directory = prepare_sub_folder(opts.output_path)
    shutil.copy(opts.config, os.path.join(opts.output_path, 'config.yaml'))
    os.makedirs(os.path.join(opts.output_path, 'scripts'), exist_ok=True)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path, "logs", model_name))
    scripts_to_save = make_dataset('./', '.py')
    for script in scripts_to_save:
        dst_file = os.path.join(opts.output_path, 'scripts', os.path.basename(script))
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copyfile(script, dst_file)

    # Load experiment setting
    config = get_config(opts.config)
    device = "cuda"

    g = Generator().to(device)

    total_params = sum(p.numel() for p in g.parameters())
    print(f'{total_params / 1000000:.4f}M total parameters in g.')

    params_g = list(g.parameters())
    opt_g = torch.optim.Adam(
        [p for p in params_g if p.requires_grad], 
        lr=config['lr'], betas=(config['beta1'], config['beta2'])
    )

    config['iterations'] = 0
    if opts.resume is not None:
        ckpt = opts.resume
        state_dict = torch.load(ckpt)
        config['iterations'] = int(state_dict['iterations'])
        g.load_state_dict(state_dict['g'])
        opt_g.load_state_dict(state_dict['opt_g'])
        print('Resume from iteration %d' % config['iterations'])

    train_loader, _ = get_data_loaders_enhance(config)

    train(
        opts, config,
        g, opt_g, 
        train_loader, 
        device, checkpoint_directory, image_directory
        )
