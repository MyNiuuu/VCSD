import os
import yaml
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data import ImageFolder2D, ImageFolderEn
from torch.utils.data import DataLoader


def make_dataset(dir, key):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(key):
                path = os.path.join(root, fname)
                if 'experiment' not in path:
                    images.append(path)
    
    return images


def get_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    height = conf['crop_image_height']
    width = conf['crop_image_width']

    train_loader = get_2D_loader_folder(conf['dataroot'], conf['noiseroot'], 
        batch_size, True, height, width, num_workers, True)
    
    return train_loader, None


def get_2D_loader_folder(input_folder, noiseroot, batch_size, train,
                           height=256, width=256, num_workers=4, crop=True, distributed=False):
    dataset = ImageFolder2D(input_folder, noiseroot, None, False, train, height, width, crop)
    print('The length of the dataset is:', len(dataset))
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, 
        drop_last=True, num_workers=num_workers,
        sampler=data_sampler(dataset, shuffle=True, distributed=distributed),
        pin_memory=True
    )
    return loader


def data_sampler(dataset, shuffle, distributed=False):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory, exist_ok=True)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory, exist_ok=True)
    return checkpoint_directory, image_directory


def viz_nolight(output, gt, vis, noisy_vis, nir, noisy_nir, wavelength, 
            ref_LED, scotopic, origin_led_spectral, led_spectral, pic_path):
    all_imgs = []

    for i in range(min(output.shape[0], 4)):
        b_nir = nir[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_noisy_nir = noisy_nir[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_vis = vis[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_noisy_vis = noisy_vis[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_output = output[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_gt = gt[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        
        style.use('seaborn-bright')
        plt.xlabel('wavelength')
        plt.ylabel('intensity')
        plt.grid(True)
        markes = ['-o', '-s', '-^', '-p', '-^', '-v', '-p', '-d', '-h', '-2', '-8', '-6']
        plt.plot(wavelength.detach().cpu(), scotopic.detach().cpu() / torch.tensor(1694), markes[2])
        plt.plot(wavelength.detach().cpu(), ref_LED.detach().cpu() / torch.max(ref_LED.detach().cpu()), markes[3])
        plt.plot(wavelength.detach().cpu(), origin_led_spectral.detach().cpu(), markes[4])
        plt.plot(wavelength.detach().cpu(), led_spectral.detach().cpu(), markes[5])
        plt.legend([
            "Mesopic Vision $V^b_M(\lambda)$",
            "Reference LED $\Phi^a$",
            "Origin Temp LED $\Phi^b$",
            "Modified Temp LED $\Phi^\hat{b}$"
        ])
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        curve_image = Image.frombytes("RGBA", (w, h), buf.tobytes()).convert('RGB')
        curve_image = np.array(curve_image)
        curve_image = curve_image
        plt.cla()
        W, H, C = curve_image.shape
        w, h, c = b_output.shape
        curve_image = cv2.resize(curve_image, (int(H * w / W), w))

        all_imgs.append(
            np.hstack(
                [b_vis, b_noisy_vis, b_nir, b_noisy_nir, curve_image, b_output, b_gt]
            )
        )

    Image.fromarray(np.vstack(all_imgs).astype(np.uint8)).save(pic_path)




def viz_enhance(output, gt, vis, nir, pic_path):
    all_imgs = []

    for i in range(min(output.shape[0], 4)):

        b_nir = nir[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_vis = vis[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_output = output[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        b_gt = gt[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
        all_imgs.append(
            np.hstack(
                [b_vis, b_nir, b_output, b_gt]
            )
        )

    Image.fromarray(np.vstack(all_imgs).astype(np.uint8)).save(pic_path)


def get_data_loaders_enhance(conf):

    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    height = conf['crop_image_height']
    width = conf['crop_image_width']

    train_loader = get_2D_loader_folder_enhance(conf['dataroot'], 
        batch_size, True, height, width, num_workers, True)
    
    return train_loader, None


def get_2D_loader_folder_enhance(input_folder, batch_size, train,
    height=256, width=256, num_workers=4, crop=True):
    # print(name)
    # assert False
    dataset = ImageFolderEn(input_folder, None, False, train, height, width, crop)
    print('The length of the dataset is:', len(dataset))
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, 
        drop_last=True, num_workers=num_workers,
        pin_memory=True
    )
    # prefetcher = data_prefetcher(loader)
    return loader