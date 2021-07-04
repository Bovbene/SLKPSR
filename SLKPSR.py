# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:39:37 2021

@author: 月光下的云海
"""


from MODELS.CKP import CKP 
from MODELS.SLSR import SLSR 

def slkpsr(lr_image,scale):
    ckp = CKP(scale = scale,epoch = 0)
    slsr = SLSR(scale = scale,epoch = 0)
    p_ker = ckp.test(lr_image)
    sr_image = slsr.test(lr_image,p_ker)
    return sr_image

# from DLL.Valuation import psnr
# from PIL import Image
# import argparse
# import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument('--scale', type=int, default = 4, help='Scale Factor')
# args = parser.parse_args()

# if __name__ == '__main__':
    
#     hr = Image.open('./DATABASE/Set5/butterfly_GT.bmp')
    
#     lr = hr.resize((hr.size[0]//args.scale,hr.size[1]//args.scale), Image.BICUBIC )
    
#     lr_image = np.array(lr)
#     hr_image = np.array(hr)
    
#     sr_image = slkpsr(lr_image,args.scale)
    
#     print("The PSNR of SR image is: "+str(psnr(sr_image,hr_image)))
#     Image.fromarray(sr_image).show()
    
