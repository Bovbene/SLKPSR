# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:22:00 2020

@author: 月光下的云海
"""

from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
from scipy.signal import convolve2d
from skimage.measure import compare_psnr,compare_ssim
import math

def rgb2ycbcr(img):
    y = 16 + (65.481 * img[:, :, 0]) + (128.553 * img[:, :, 1]) + (24.966 * img[:, :, 2])
    return y / 255

def PSNR(target, ref, scale):
    target_data = np.array(target, dtype=np.float32)
    ref_data = np.array(ref, dtype=np.float32)

    target_y = rgb2ycbcr(target_data)
    ref_y = rgb2ycbcr(ref_data)
    diff = ref_y - target_y

    shave = scale
    diff = diff[shave:-shave, shave:-shave]

    mse = np.mean((diff / 255) ** 2)
    if mse == 0:
        return 100

    return -10 * math.log10(mse)


'''============================================================================
FUNCTION:psnr
FEATURE: 计算两个图像间的psnr值
INPUTS:
       img1----------低分图片
       img2----------高分图像
OUTPUTS:
   psnr值
============================================================================'''  
def psnr_numpy(img1,img2):
    mse = np.mean((img1/1.0 - img2/1.0)**2)
    if(mse<1e-1):
        return 100
    return 10*np.log10(255.0**2/mse)

def psnr(img1,img2):
    return compare_psnr(img1,img2,255)

def ssim(img1,img2):
    return compare_ssim(img1,img2,multichannel = True)

'''这个我也不知是啥'''
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
'''这个我也不知是啥'''
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

'''============================================================================
FUNCTION:ssim
FEATURE: 计算两个图像间的ssim值
INPUTS:
       img1----------低分图片
       img2----------高分图像
OUTPUTS:
   ssim值
============================================================================'''  
def ssim_numpy(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    return np.mean(np.mean(ssim_map))

def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse
    
def mae(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae    

#def ssim(y_pred,y_true):
#    u_true = np.mean(y_true)
#    u_pred = np.mean(y_pred)
#    var_true = np.var(y_true)
#    var_pred = np.var(y_pred)
#    std_true = np.sqrt(var_true)
#    std_pred = np.sqrt(var_pred)
#    c1 = np.square(0.01*7)
#    c2 = np.square(0.03*7)
#    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#    return ssim/denom






