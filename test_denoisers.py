#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:56:48 2022

@author: electron
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from pnp import gaussian_filter, bilateral_filter
from bm3d import bm3d_rgb
from KAIR.denoise_dncnn import denoise_dncnn
from MPRNet.denoise_mprnet import denoise_mprnet

import torchvision.transforms.functional as TF
import torch

def PSNR(original, compressed):
    # https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

noise_level_img = 60

img_name = '/home/electron/sources/kodak_dataset/kodim15.png'
img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img/255.)
imgn = img + np.random.normal(0, noise_level_img/255., img.shape)
imgn[imgn>1] = 1

out1 = gaussian_filter(imgn, 1)
out2 = bilateral_filter(imgn,1)
out3 = bm3d_rgb(imgn, 0.1)
out4 = denoise_dncnn(imgn)
out4 = np.float32(out4)/255.

imgt = TF.to_tensor(img).unsqueeze(0)
imgt = imgt + (noise_level_img/255)*torch.randn_like(imgt)
out5 = denoise_mprnet(imgt)
out5 = np.float32(out5)/255.
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(img)
axarr[0,1].imshow(out1)
axarr[1,0].imshow(out2)
axarr[1,1].imshow(out3)

print(f'PSNR for Noisy image = {PSNR(img, imgn)}')
print(f'PSNR for Gaussian filter = {PSNR(img, out1)}')
print(f'PSNR for Bilateral filter = {PSNR(img, out2)}')
print(f'PSNR for BM3D = {PSNR(img, out3)}')
print(f'PSNR for DNCNN = {PSNR(img, out4)}')
print(f'PSNR for MPRNet = {PSNR(img, out5)}')


imageio.imwrite('img.png', img)
imageio.imwrite('imgn.png', imgn)
imageio.imwrite('out_gauss.png', out1)
imageio.imwrite('out_bilateral.png', out2)
imageio.imwrite('out_bm3d.png', out3)
imageio.imwrite('out_dncnn.png', out4)
imageio.imwrite('out_mprnet.png', out5)