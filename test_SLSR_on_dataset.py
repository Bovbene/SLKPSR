# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:28:14 2021

@author: 月光下的云海
"""

from MODELS.SLSR import SLSR as model
import argparse
from DLL.utils import fspecial,GetBlurMtx
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
from DLL.Valuation import psnr,ssim
from time import time
import cv2 as cv

def test_SLSR_on_dataset(scale,path):
    
    data_set = os.path.split(path)[0]
    data_set = os.path.split(data_set)[-1]
    
    if not os.path.exists('./RESULT/SLSRx{}/'.format(scale)+data_set):
        os.makedirs('./RESULT/SLSRx{}/'.format(scale)+data_set)
    
    slsr = model(scale = scale,epoch = 0)
    blur_kernel = fspecial(kernel_size = 17,sigma = 4)
    iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
    pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(scale)
    sess = tf.Session(config = slsr.config)
    with gfile.FastGFile(pbPath,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name = '')
    LR_tensor = sess.graph.get_tensor_by_name("images:0")
    HR_tensor = sess.graph.get_tensor_by_name("add_100:0")
    #H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
    H = sess.graph.get_tensor_by_name("H:0")
    val_img_paths = glob(path)
    avg_psnr = 0
    for p in val_img_paths:
        fn = os.path.split(p)[-1]
        print("Testing on "+ fn + ' ... ...')
        t = time()
        fn = os.path.splitext(fn)[0]
        hr = cv.imread(p)
        hr = Image.fromarray(hr.astype(dtype=np.uint8))
        
        lr = hr.resize((hr.size[0]//scale,hr.size[1]//scale), Image.BICUBIC)
        hr = np.array(hr)
        lr = np.array(lr).astype(np.uint8)
        lr = lr.reshape((1,)+lr.shape)
        #hr,lr = get_image(p, scale, None)
        res = sess.run(HR_tensor, feed_dict={LR_tensor: lr / 255.0, H:iH})
        
        res = res[0]*255.0
        res = res.clip(min = 0, max = 255)
        #res = res.astype(np.uint8)
        ipsnr,issim = psnr(hr,res),ssim(hr,res)
        print("Time Elapsed:", time()-t,end = ' ')
        print('The PSNR:{:.4f} and SSIM:{:.4f}'.format(ipsnr,issim))
        avg_psnr += ipsnr
        #Image.fromarray(res).save('./RESULT/SLSRx{}/'.format(scale) +'/'+data_set+'/'+ fn +'_{:.4f}_{:.4f}.png'.format(ipsnr,issim))
        cv.imwrite('./RESULT/SLSRx{}/'.format(scale) +'/'+data_set+'/'+ fn +'_{:.4f}_{:.4f}.png'.format(ipsnr,issim),res)
    sess.close()
    print("Avg. :",avg_psnr/len(val_img_paths))

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default = 4, help='Scale Factor')
parser.add_argument('--path', type = str, default = './DATABASE/Set5/*.bmp')
args = parser.parse_args()

if __name__ == '__main__':
    
    test_SLSR_on_dataset(args.scale,args.path)
    
    