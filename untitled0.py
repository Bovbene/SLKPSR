# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:16:07 2021

@author: 月光下的云海
"""

from glob import glob
import cv2 as cv
import os
from DLL.Valuation import PSNR as psnr
from DLL.Valuation import ssim

def get_name(p):
    fn = os.path.split(p)[-1]
    fn = os.path.splitext(fn)[0]
    return fn

if __name__ == '__main__':
    data_set = 'Set5'
    scale = 4
    sr_fnli = glob('./RESULT/SML-SRx{}/'.format(scale)+data_set+'/*.png')
    hr_fnli = glob('./DATABASE/'+data_set+'/*.bmp')
    
    for sr_fn,hr_fn in zip(sr_fnli,hr_fnli):
        fn = get_name(hr_fn)
        # if get_name(sr_fn) != get_name(hr_fn):
        #     raise AttributeError("Match Error!")
        
        sr = cv.imread(sr_fn)
        hr = cv.imread(hr_fn)
        
        if sr.shape != hr.shape:
            sr = cv.resize(sr,(hr.shape[1],hr.shape[0]),interpolation = cv.INTER_CUBIC)
        
        ipsnr,issim = psnr(hr,sr,scale = 4),ssim(hr,sr)
        print(ipsnr,issim,sr_fn)
        cv.imwrite('./RESULT/SML-SRx{}/'.format(scale) +'/'+data_set+'/'+ fn +'_{:.4f}_{:.4f}.png'.format(ipsnr,issim),sr)
        os.remove(sr_fn)
        

