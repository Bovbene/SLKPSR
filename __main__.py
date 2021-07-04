# -*- coding: utf-8 -*-
"""
Created on Sat May  1 13:35:34 2021

@author: 月光下的云海
"""

from DLL.Valuation import psnr as psnr
import cv2 as cv
# from DLL.utils import Show
import argparse
import numpy as np
from PIL import Image
from DLL.utils import DegradeFilter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = 'SLKPSR', help=['SrOp','CKP','SLSR','SLKPSR','SRCNN','VDSR'])
parser.add_argument('--is_train', type=bool, default = False, help='Train or test.')
parser.add_argument('--scale', type=int, default = 1, help='Scale Factor')
parser.add_argument('--epoch', type = int, default = 50, help = "The training epochs.")
parser.add_argument('--f1', type = str, default = './DATABASE/DIV2K/*.png')
parser.add_argument('--f2', type = str, default = './DATABASE/MANGA109/*.bmp')
args = parser.parse_args()

if __name__ == '__main__':
    
    exec( "from MODELS.%s import %s as Model"%(args.model,args.model) )
    model = Model(scale = args.scale,epoch = args.epoch)
    if args.is_train:
        print( model.train(args.f1,args.f2) )
    else:
        model.eval(args.f2)
        
        # hr_image = cv.imread('./DATABASE/DIV2K/0005.png')
        # hr_image = hr_image[:512,:512]
        # import scipy.io as scio
        # real_ker = scio.loadmat('./DATABASE/DIV2K_BlurKernel/0005/0005_kernel_x2.mat')
        # real_ker = np.array(real_ker['Kernel'])
        # lr_image = DegradeFilter(hr_image,real_ker)
        # pre_ker = model.test(lr_image)
        '''
        hr_image = Image.open('./DATABASE/Set5/butterfly_GT.bmp')
        #lr_image = hr_image.resize((hr_image.size[0]//args.scale,hr_image.size[1]//args.scale), Image.BICUBIC)
        import scipy.io as scio
        real_ker = scio.loadmat('./DATABASE/DIV2K_BlurKernel/0005/0005_kernel_x2.mat')
        real_ker = np.array(real_ker['Kernel'])
        
        lr_image = np.array(hr_image)
        lr_image = DegradeFilter(lr_image,real_ker)
        res = model.test(lr_image)
        
        try:
            print("The PSNR of SR image is: "+str(psnr(res,np.array(hr_image))))
            Image.fromarray(res).show()
        except ValueError:
            print(res)

        '''
    
    