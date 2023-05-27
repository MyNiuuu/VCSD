# VCSD

Welcome! This is the official PyTorch implementation for our paper: 

🤖 [CVPR2023] [**Visibility Constrained Wide-band Illumination Spectrum Design for Seeing-in-the-Dark**](https://openaccess.thecvf.com/content/CVPR2023/papers/Niu_Visibility_Constrained_Wide-Band_Illumination_Spectrum_Design_for_Seeing-in-the-Dark_CVPR_2023_paper.pdf)

by Muyao Niu, Zhuoxiao Li, Zhihang Zhong, and Yinqiang Zheng.

🔗 [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Niu_Visibility_Constrained_Wide-Band_Illumination_Spectrum_Design_for_Seeing-in-the-Dark_CVPR_2023_paper.pdf), [arxiv](https://arxiv.org/abs/2303.11642)

> In this paper, we try to robustify NIR2RGB translation by designing the optimal spectrum of auxiliary illumination in the wide-band VIS-NIR range, while keeping visual friendliness. Our core idea is to quantify the visibility constraint implied by the human vision system and incorporate it into the design pipeline. By modeling the formation process of images in the VIS-NIR range, the optimal multiplexing of a wide range of LEDs is automatically designed in a fully differentiable manner, within the feasible region defined by the visibility constraint. We also collect a substantially expanded VIS-NIR hyperspectral image dataset for experiments by using a customized 50-band filter wheel. Experimental results show that the task can be significantly improved by using the optimized wide-band illumination than using NIR only.


<img src="./documents/teaser.png"/>


If you find this work interesting, please do not hesitate to leave a ⭐!

## Environment Setup

```
conda create -n vcsd
conda activate vcsd
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install tqdm
pip install tensorboardX
pip install pyyaml
pip install opencv-python
pip install matplotlib
pip install scikit-image
pip install h5py
```

## Data Preparation

Our original hyperspectral dataset can be downloaded [here](https://drive.google.com/file/d/1f-MgXjHil7-SibIj1uhZTPhq5LzdHcHH/view?usp=share_link).

Our dataset contains 74 hypersprectral scenes and the corresponding synthesized RGB images using camera GS3-U3-15S5C. Each scene has a spatial resolution of $1936 \times 1096$, and a spectral resolution of $50$ ($400nm \sim 890nm$). 

Note that to load data faster during the curve design phase, we divide each scene into the spatial resolution of $484 \times 274$. If you wish to run the curve design code of our method, it is recommended to also download the corresponding divided data [here](https://drive.google.com/file/d/10Ms3zvupe8NSWvu9bzsjkC9SGNRI4FNi/view?usp=share_link). 

The sampled noise pattern from GS3-U3-15S5C can be downloaded [here](https://drive.google.com/file/d/14zvz-LT-6k2dHVgyuaE5jmOwwSx6zpMS/view?usp=share_link).

After downloading the data, unzip it into `./`.

## Training
Run the following commands to design the optimal curve under the visibility constraint:
```
CUDA_VISIBLE_DEVICES=0 \
python design.py \
--config configs/config_design.yaml \
--output_path EXP_design/vcsd
```

Then synthesize the assistant images using the designed optimal LED curve by running:
```
python synthesize_with_curve.py \
--root ./dataset \
--save_root ./dataset_pics \
--ckpt_path PATH_TO_CKPT \
--camera ./spectrum_data/GS3-U3-15S5C_420_890.npy \
--whitelight ./spectrum_data/white_led_61.npy
```

Finally, run the following commands to train the enhancement model using the assistant images
```
CUDA_VISIBLE_DEVICES=0 \
python enhance.py \
--config configs/config_enhance.yaml \
--output_path EXP_enhance/vcsd
```


## Testing
Run
```
CUDA_VISIBLE_DEVICES=1 \
python test.py \
--config configs/config_enhance.yaml \
--save_to test_results \
--resume PATH_TO_CKPT
```
to test the enhancement result.

## Citation

If you find this repo useful, please consider citing:
```
@InProceedings{Niu_2023_CVPR,
    author    = {Niu, Muyao and Li, Zhuoxiao and Zhong, Zhihang and Zheng, Yinqiang},
    title     = {Visibility Constrained Wide-Band Illumination Spectrum Design for Seeing-in-the-Dark},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {13976-13985}
}
```
