# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:18:59 2021

@author: 月光下的云海
"""
import os
import time
from PIL import Image
import glob
import numpy as np
from skimage.measure import compare_ssim as ssim
from MODELS.CSCN import SCN,modcrop
from MODELS.CSCN import shave as Shave
import argparse

def evalimg(im_h_y, im_gt, shave=0):
    im_h_y_uint8 = np.rint( np.clip(im_h_y, 0, 255),dtype = np.float64)
    im_gt_y_uint8 = np.rint( np.clip(im_gt, 0, 255),dtype = np.float64)
    diff = im_h_y_uint8 - im_gt_y_uint8
    if shave>0:
        diff = Shave(diff, [shave, shave])
    res = {}
    res['ssim'] = ssim(im_h_y_uint8,im_gt_y_uint8,multichannel=True)
    res['rmse'] = np.sqrt((diff**2).mean())
    res['psnr'] = 20*np.log10(255.0/res['rmse'])
    #res['psnr'] = psnr(im_h_y_uint8,im_gt_y_uint8)
    return res

def test_CSCN_on_dataset(scale,path):
    #input with two images
    #IMAGE_FILE='./data/slena.bmp'
    IMAGE_FILE=''
    IMAGE_GT_FILE=path
    """
    # input with ground truth images only (Matlab required)
    IMAGE_FILE=''
    IMAGE_GT_FILE='./data/Set5/*.bmp'
    """
    
    data_set = os.path.split(path)[0]
    data_set = os.path.split(data_set)[-1]
    
    MODEL_FILE=['./TRAINED_MODEL/CSCN_x2_52.p', './TRAINED_MODEL/CSCN_x2_310.p']
    UP_SCALE=scale
    up_flag = int(UP_SCALE)
    SHAVE=1 #set 1 to be consistant with SRCNN
    
    if not os.path.exists("./RESULT/CSCNx{}/".format(UP_SCALE)+data_set):
        os.makedirs("./RESULT/CSCNx{}/".format(UP_SCALE)+data_set)
    
    # load inputs
    im_gt = []
    files_gt = glob.glob(IMAGE_GT_FILE)
    for f in files_gt:
        #print 'loading', f
        im = np.array(Image.open(f))
        im = modcrop(im, UP_SCALE).astype(np.float32)
        im_gt += [im]


    im_l,fn_li = [],[]
    if len(IMAGE_FILE)>0:
        assert(len(im_gt)==1)
        im_l = [np.array(Image.open(IMAGE_FILE)).astype(np.float32)]
    else: #down scale from ground truth using Matlab
        for f in files_gt:
            fn = os.path.split(f)[-1]
            fn = os.path.splitext(fn)[0]
            hr_im = Image.open(f)
            lr_im = hr_im.resize((hr_im.size[0]//UP_SCALE,hr_im.size[1]//UP_SCALE),Image.BICUBIC)
            lr_im = np.array(lr_im).astype(np.float32)
            im_l += [lr_im]
            fn_li += [fn]
            
            
    if(UP_SCALE == 3 and len(IMAGE_FILE) == 0):
        im_l = []
        for f in files_gt:
            hr_im = Image.open(f)
            hr_im = hr_im.resize((hr_im.size[0]//UP_SCALE*UP_SCALE,hr_im.size[1]//UP_SCALE*UP_SCALE),Image.BICUBIC)
            lr_im = hr_im.resize((hr_im.size[0]//UP_SCALE,hr_im.size[1]//UP_SCALE))
            lr_im = lr_im.resize((hr_im.size[0]//2,hr_im.size[1]//2))
            lr_im = np.array(lr_im).astype(np.float32)
            im_l += [lr_im]
            
        UP_SCALE = 2
    
    #upscaling
    #sr = Bicubic()
    sr = SCN(MODEL_FILE)
    res_all = []
    res_psnr = []
    res_ssim = []
    for i in range(len(im_l)):
        t=time.time();
        im_h, im_h_y=sr.upscale(im_l[i], UP_SCALE)
        t=time.time()-t;
        print('time elapsed:', t)

        # evaluation
        if SHAVE==1:
            shave = round(UP_SCALE)
        else:
            shave = 0
        res = evalimg(im_h, im_gt[i], shave)
        res_all += [res]
        res_psnr.append(res['psnr'])
        res_ssim.append(res['ssim'])
        print('evaluation against {}, rms={:.4f}, psnr={:.4f}, ssim={:.4f}'.format(files_gt[i], res['rmse'], res['psnr'],res['ssim']))

        # save
        # img_name = os.path.splitext(os.path.basename(files_gt[i]))[0]
        Image.fromarray(np.rint(im_h).astype(np.uint8)).save('./RESULT/CSCNx{}/'.format(up_flag)+data_set+'/'+fn_li[i]+'_{:.4f}_{:.4f}.png'.format(res['psnr'], res['ssim']))

    print('mean PSNR:', np.array([_['psnr'] for _ in res_all]).mean())

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default = 4, help='Scale Factor')
parser.add_argument('--path', type = str, default = './DATABASE/MANGA109/*.bmp')
args = parser.parse_args()

if __name__ == '__main__':
    test_CSCN_on_dataset(args.scale,args.path)
    
    