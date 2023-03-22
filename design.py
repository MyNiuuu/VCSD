import argparse
import os
import csv
import torch
import torch.nn as nn
import shutil
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch import nn
from tqdm import tqdm
import tensorboardX
from model import VCSD as Generator
from loss import PerceptualLoss
from util import sample_data, get_config, prepare_sub_folder, \
    get_data_loaders, viz_nolight, make_dataset


def train(
        opts, config, wavelength, 
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

        spectral, gt, noise = next(train_loader)
        spectral, gt, noise = spectral.to(device), gt.to(device), noise.to(device)
        
        opt_g.zero_grad()
        output, vis, nir, noisy_vis, noisy_nir, wavelength, \
            ref_LED, led_intersect, xi, ksi_vis, ksi_nir, scotopic, SP_ratio, \
                origin_led_spectral, led_spectral, ksi_coefficient = g(
                    spectral, noise
                )

        loss_G_percep = percep(output, gt) * opts.perceplamb
        
        loss_G = loss_G_percep
        loss_G.backward()
        opt_g.step()
        
        if i % 20000 == 1 and i > 10:
            for param_group in opt_g.param_groups:
                temp_lr = param_group['lr']
                param_group['lr'] = temp_lr * 0.1

        description = f"percep_loss: {loss_G_percep:.4f}"
        pbar.set_description((description))
        train_writer.add_scalar('percep_loss', loss_G_percep.item(), i)
        if i % config['image_save_iter'] == 0:
            viz_nolight(
                output, gt, vis, noisy_vis, nir, noisy_nir, 
                wavelength, ref_LED, scotopic, origin_led_spectral, led_spectral,
                os.path.join(image_directory, str(i).zfill(7) + '_train_pic.png')
            )

        if i % config['snapshot_save_iter'] == 0:
            print("curve:", led_spectral)
            print("ksi_coefficient:", ksi_coefficient)
            print('ksi_vis:', ksi_vis)
            print('ksi_nir:', ksi_nir)
            save_dict = {
                "g": g.state_dict(),
                "opt_g": opt_g.state_dict(),
                "select": g.op.select,
                "curve": led_spectral,
                "ksi_coefficient": ksi_coefficient,
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
    parser.add_argument('--perceplamb', type=float, default=1, help='')
    opts = parser.parse_args()

    cudnn.benchmark = True
    setup_seed(521)

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    checkpoint_directory, image_directory = prepare_sub_folder(opts.output_path)
    shutil.copy(opts.config, os.path.join(opts.output_path, 'config.yaml'))
    os.makedirs(os.path.join(opts.output_path, 'scripts'), exist_ok=True)

    # Load experiment setting
    config = get_config(opts.config)
    device = "cuda"

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path, "logs", model_name))
    scripts_to_save = make_dataset('./', '.py')
    for script in scripts_to_save:
        dst_file = os.path.join(opts.output_path, 'scripts', os.path.basename(script))
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copyfile(script, dst_file)

    camera = np.load('./spectrum_data/GS3-U3-15S5C_420_890.npy')[::-1, :].copy()

    ans = []
    with open("./spectrum_data/EyeSensitivity.csv", mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line in reader: #Iterate through the loop to read line by line
            for i in range(len(line)):
                line[i] = float(line[i])
            ans.append(line)

    ans = np.array(ans)[2:, :]  # [42, 5]
    extra_wavelength = np.linspace(770, 890, 13)[:, np.newaxis]
    extra_ans = np.concatenate([extra_wavelength, np.zeros((13, 4))], axis=1)
    ans = np.concatenate([ans, extra_ans], axis=0)  # 420nm~890nm
    
    wavelength = ans[:, 0]
    scotopic = ans[:, 4]
    photopic = ans[:, 3]

    bases = np.load('./spectrum_data/led_31allcut_420_1000.npy')[:-5, :-11]
    ref_LED = bases[13]

    print('shape of CSS: ', camera.shape)
    print('shape of EyeSensitivity: ', ans.shape)
    print('shape of LED bases:', bases.shape)

    g = Generator(
        wavelength, camera, scotopic, photopic, 
        ref_LED, bases, config['gain']
    ).to(device)

    total_params = sum(p.numel() for p in g.parameters())
    print(f'{total_params / 1000000:.4f}M total parameters in g.')

    params_g = list(g.parameters())
    opt_g = torch.optim.Adam(
        [p for p in params_g if p.requires_grad], 
        lr=config['lr'], betas=(config['beta1'], config['beta2']), 
    )

    config['iterations'] = 0
    if opts.resume is not None:
        ckpt = opts.resume
        state_dict = torch.load(ckpt)
        config['iterations'] = int(state_dict['iterations'])
        g.load_state_dict(state_dict['g'])
        opt_g.load_state_dict(state_dict['opt_g'])
        print('Resume from iteration %d' % config['iterations'])

    train_loader, _ = get_data_loaders(config)

    train(
        opts, config, wavelength,
        g, opt_g,
        train_loader, 
        device, checkpoint_directory, image_directory
        )
