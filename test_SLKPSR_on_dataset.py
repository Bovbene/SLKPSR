# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:26:01 2021

@author: 月光下的云海
"""

import argparse
from MODELS.CKP import CKP 
from MODELS.SLSR import SLSR 
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from glob import glob
from time import time
from PIL import Image
import numpy as np
from DLL.utils import GetBlurMtx
from DLL.Valuation import psnr,ssim

def test_SLKPSR_on_dataset(scale,path):
    if not os.path.exists('./RESULT/SLKPSRx{}/'.format(scale)):
        os.makedirs('./RESULT/SLKPSRx{}/'.format(scale))
    
    data_set = os.path.split(path)[0]
    data_set = os.path.split(data_set)[-1]
    
    ckp = CKP(scale = scale,epoch = 0)
    pbPath = "./TRAINED_MODEL/CKP.pb"
    sess_ckp = tf.Session(config = ckp.config)
    with gfile.FastGFile(pbPath,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess_ckp.graph.as_default()
        tf.import_graph_def(graph_def,name = '')
    LR_tensor1 = sess_ckp.graph.get_tensor_by_name("InputPh:0")
    ker_map = sess_ckp.graph.get_tensor_by_name("Predictor/Reshape:0")
    
    slsr = SLSR(scale = scale,epoch = 0)
    pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(scale)
    sess_slsr = tf.Session(config = slsr.config)
    with gfile.FastGFile(pbPath,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess_slsr.graph.as_default()
        tf.import_graph_def(graph_def,name = '')
    LR_tensor2 = sess_slsr.graph.get_tensor_by_name("images:0")
    HR_tensor = sess_slsr.graph.get_tensor_by_name("add_100:0")
    H = sess_slsr.graph.get_tensor_by_name("H:0")
    #H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
    
    val_img_paths = glob(path)
    
    for p in val_img_paths:
    
        fn = os.path.split(p)[-1]
        print("Testing on "+ fn + ' ... ...')
        t = time()
        fn = os.path.splitext(fn)[0]
        hr = Image.open(p)
        lr = hr.resize((hr.size[0]//scale,hr.size[1]//scale), Image.BICUBIC)
        hr = np.array(hr)
        lr = np.array(lr)
        lr = lr.reshape((1,)+lr.shape)
        
        blur_kernel = sess_ckp.run(ker_map, feed_dict={LR_tensor1: lr})
        blur_kernel = blur_kernel[0]
        iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        
        res = sess_slsr.run(HR_tensor, feed_dict={LR_tensor2: lr/255.0,H:iH})
        
        res = res[0]
        res = res.clip(min = 0, max = 255)
        res = res.astype(np.uint8)
        
        ipsnr,issim = psnr(hr,res),ssim(hr,res)
        print("Time Elapsed:", time()-t,end = ' ')
        print('The PSNR:{:.4f} and SSIM:{:.4f}'.format(ipsnr,issim))
        Image.fromarray(res).save('./RESULT/SLKPSRx{}/'.format(scale) +'/'+data_set+'/'+ fn +'{:.4f}_{:.4f}.png'.format(ipsnr,issim))
        
    sess_slsr.close()
    sess_ckp.close()

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default = 4, help='Scale Factor')
parser.add_argument('--path', type = str, default = './DATABASE/Set5/*.bmp')
args = parser.parse_args()

if __name__ == '__main__':
    
    test_SLKPSR_on_dataset(args.scale,args.path)
    
    