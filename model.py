import torch
import torch.nn as nn
import numpy as np

from unet import UnetGenerator


class NoiseModel(nn.Module):
    def __init__(self, gain):
        super(NoiseModel, self).__init__()
        self.gain = gain
       
    def get_noise(self, img, ksi):
        img_255 = img * 4095.
        poisson = torch.distributions.Poisson((img_255 * ksi) / self.gain)
        noise_img = poisson.sample().to(img.device)
        noise = noise_img - img_255
        return noise / 4095.

    def forward(self, image, ksi, noise_pattern):
        noise = self.get_noise(image, ksi)
        return (image + noise) * self.gain + noise_pattern


class MultiplexCurve(nn.Module):
    def __init__(self, bases):
        super(MultiplexCurve, self).__init__()
        self.register_buffer('spectral_base', torch.tensor(bases).float())
        self.select = nn.Parameter(torch.rand(self.spectral_base.shape[0], 1))
      
    def forward(self):
        for i in self.select:
            i.data.clamp_(1e-6, 1)
        curve = self.spectral_base.permute(1, 0) @ self.select
        return curve


class OptimalSpectral(nn.Module):
    def __init__(self, wavelength, camera_responce, scotopic, 
        photopic, reference_led, bases):
        super(OptimalSpectral, self).__init__()
        self.register_buffer('wavelength', torch.tensor(wavelength).float())
        self.register_buffer('camera_responce', torch.tensor(camera_responce).reshape(1, 3, wavelength.shape[0], 1, 1).float())
        self.register_buffer('scotopic', torch.tensor(scotopic).float())
        self.register_buffer('photopic', torch.tensor(photopic).float())
        self.register_buffer('reference_led', torch.tensor(reference_led).float())
        self.register_buffer('bases', torch.tensor(bases).float())

        self.select = nn.Parameter(torch.randn(bases.shape[0], 1))
        
        ref_intersect, _ = self.calculate_intersect_area(self.reference_led)
        
        self.ref_intersect = ref_intersect

        self.register_buffer('one', torch.tensor(1).float())

        led_num = bases.shape[0]
        bases_intersect = np.zeros((led_num))
        for i in range(led_num):
            bases_intersect[i] = self.calculate_intersect_area(self.bases[i])[0]
        self.register_buffer('bases_intersect', torch.tensor(bases_intersect).float())

    def forward(self, image_responce, given_led=None):
        b = image_responce.shape[0]
        if given_led is not None:
            origin_led_spectral = given_led
        else:
            for i in self.select:
                i.data.clamp_(1e-6, 1)
            origin_coefficient = self.select / torch.sum(self.select) * 2
            origin_led_spectral = self.bases.permute(1, 0) @ origin_coefficient
        
        ksi, led_intersect, SP_ratio = self.calculate_ksi(origin_led_spectral)

        ksi_coefficient = torch.where(self.bases_intersect > 0, ksi * origin_coefficient.squeeze(), origin_coefficient.squeeze())
        led_spectral = self.bases.permute(1, 0) @ ksi_coefficient

        led_spectral = led_spectral.reshape(1, 1, self.wavelength.shape[0], 1, 1)
        image_responce = image_responce.unsqueeze(1)
        
        vis_led_spectral = led_spectral[:, :, :28]
        vis_camera_response = self.camera_responce[:, :, :28]
        vis_image_responce = image_responce[:, :, :28]

        nir_led_spectral = led_spectral
        nir_camera_response = self.camera_responce
        nir_image_responce = image_responce

        origin_vis_value = torch.sum(origin_led_spectral[:28])
        origin_nir_value = torch.sum(origin_led_spectral)

        vis_value = torch.sum(vis_led_spectral)
        nir_value = torch.sum(nir_led_spectral)

        ksi_vis = vis_value / (origin_vis_value + 1e-6)
        ksi_nir = nir_value / (origin_nir_value + 1e-6)

        vis_image = torch.sum(vis_led_spectral * vis_camera_response * vis_image_responce, dim=2)
        vis_batch_max = torch.max(torch.max(torch.max(vis_image, dim=3)[0],dim=2)[0],dim=1)[0].reshape(b, 1, 1, 1)
        vis_image = vis_image / (vis_batch_max + 1e-6)

        nir_image = torch.sum(nir_led_spectral * nir_camera_response * nir_image_responce, dim=2)  # [b, 3, 512, 512]
        nir_batch_max = torch.max(torch.max(torch.max(nir_image, dim=3)[0],dim=2)[0],dim=1)[0].reshape(b, 1, 1, 1)
        nir_image = nir_image / (nir_batch_max + 1e-6)

        return ksi_vis, ksi_nir, vis_image, nir_image, vis_image, nir_image, self.wavelength, self.reference_led, led_intersect, ksi, self.scotopic, SP_ratio, \
            origin_led_spectral, led_spectral.reshape(self.wavelength.shape[0]), ksi_coefficient.squeeze()
    
    def calculate_SP_ratio(self, led_spec):
        """
            led_spec: [wavelength]
        """
        return torch.sum(led_spec * self.scotopic) / (torch.sum(led_spec * self.photopic) + 1e-6)

    def calculate_intersect_area(self, led_spec):
        SP_ratio = self.calculate_SP_ratio(led_spec)
        intersect_area = torch.sum(led_spec * self.scotopic)
        # print(intersect_area)
        return intersect_area, SP_ratio
    
    def calculate_ksi(self, led_spec):
        led_spec = led_spec.squeeze()
        led_intersect, SP_ratio = self.calculate_intersect_area(led_spec)
        ksi = torch.min(self.one, self.ref_intersect / (led_intersect + 1e-6))
        return ksi, led_intersect, SP_ratio


class VCSD(nn.Module):
    def __init__(self, wavelength, camera_responce, scotopic, 
        photopic, reference_led, bases, gain):
        super(VCSD, self).__init__()
        self.op = OptimalSpectral(
            wavelength, camera_responce, scotopic, photopic, reference_led, bases
        )
        self.noise = NoiseModel(gain)
        self.net = UnetGenerator()
    
    def forward(self, image_responce, noise_pattern, given_led=None):

        ksi_vis, ksi_nir, vis, nir, low_vis, low_nir, wavelength, ref_LED, led_intersect, ksi, scotopic, SP_ratio, origin_led_spectral, \
            led_spectral, ksi_coefficient = self.op(image_responce, given_led)
        
        noisy_nir = self.noise(low_nir, ksi_nir, noise_pattern)
        noisy_vis = self.noise(low_vis, ksi_vis, noise_pattern)
        out = self.net(torch.cat([noisy_vis, noisy_nir], dim=1))

        return out, vis, nir, noisy_vis, noisy_nir, wavelength, ref_LED, led_intersect, ksi, ksi_vis, ksi_nir, scotopic, SP_ratio, origin_led_spectral, led_spectral, ksi_coefficient
