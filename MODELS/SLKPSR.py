# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:40:44 2021

@author: 月光下的云海
"""
"""===============================================================================================
The Python code of SLSR
---------------------------------------------------------------------------------------------------
Class: SrOp
Param: 	self.scale = 1
        self.lr = 1e-3
        self.batch_size = 128
        self.epochs = 2
        self.blk_size = 32
        self.save_path = './TRAINED_MODEL/'
        self.remark = "U need to enlarge the picture to the specified size first."
        self.ch = 3
        #self.ker_root = './DATABASE/DIV2K_BlurKernel/'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        tf.reset_default_graph()
---------------------------------------------------------------------------------------------------
Tip: None
---------------------------------------------------------------------------------------------------
Created on Sat May  1 21:59:06 2021
@author: 月光下的云海(西电博巍)
Version: Ultimate
==============================================================================================="""

from .BaseModel import BaseModel
import logging
import tensorflow as tf
import os
import numpy as np
from time import time,strftime,localtime
import cv2 as cv
from glob import glob
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from DLL.utils import fspecial,GetBlurMtx,setup_logger,del_file,DegradeFilter
import tensorflow.contrib.slim as slim
from DLL.Valuation import ssim,PSNR

class SLKPSR(BaseModel):
    
    def __init__(self,scale,epoch):
        
        # -- Global Config --
        self.scale = scale
        self.epochs = epoch
        self.ch = 3
        self.blk_size = 32
        
        # -- GPU Config --
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        # -- SrOp config --
        self.B = 32 #Number of resBlocks
        self.F = 256 #Number of filters
        self.batch_size = 16
        self.scaling_factor = 0.1
        self.PS = 3 * (self.scale*self.scale) #channels x scale^2
        self.save_path = './TRAINED_MODEL/'
        
        tf.reset_default_graph()
        #self.sess = tf.Session(config=self.config)
        self.sess_slsr = tf.Session(config = self.config)
        self.xavier = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.constant_initializer(value=0.0)
        self.build_slsr()
        #self.build_srop()
        self.build_ckp()
        
    
    def build_slsr(self):
        pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(self.scale)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess_slsr.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        self.LR_tensor2 = self.sess_slsr.graph.get_tensor_by_name("IteratorGetNext:0")
        self.HR_tensor = self.sess_slsr.graph.get_tensor_by_name("add_100:0")
        self.H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
        
    
    def build_ckp(self):
        pbPath = "./TRAINED_MODEL/CKP.pb"
        self.sess_ckp = tf.Session(config = self.config)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess_ckp.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        self.LR_tensor1 = self.sess_ckp.graph.get_tensor_by_name("InputPh:0")
        self.ker_map = self.sess_ckp.graph.get_tensor_by_name("Predictor/Reshape:0")
        
        
    def build_srop(self):
        
        # -- Filters & Biases --
        self.resFilters = list()
        self.resBiases = list()
        for i in range(self.B*2):
            self.resFilters.append( tf.get_variable("SrOp/resFilter%d" % (i), shape=[3,3,self.F,self.F], initializer=self.xavier))
            self.resBiases.append(tf.get_variable(name="SrOp/resBias%d" % (i), shape=[self.F], initializer=self.bias_initializer))
        self.filter_one = tf.get_variable("SrOp/resFilter_one", shape=[3,3,3,self.F], initializer=self.xavier)
        self.filter_two = tf.get_variable("SrOp/resFilter_two", shape=[3,3,self.F,self.F], initializer=self.xavier)
        self.filter_three = tf.get_variable("SrOp/resFilter_three", shape=[3,3,self.F,self.PS], initializer=self.xavier)
        
        self.bias_one = tf.get_variable(shape=[self.F], initializer=self.bias_initializer, name="SrOp/BiasOne")
        self.bias_two = tf.get_variable(shape=[self.F], initializer=self.bias_initializer, name="SrOp/BiasTwo")
        self.bias_three = tf.get_variable(shape=[self.PS], initializer=self.bias_initializer, name="SrOp/BiasThree")
        self.sess.run(tf.global_variables_initializer())
        
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        ckpt_name = self.save_path + "SrOp_x{}/model.ckpt".format(self.scale)
        saver.restore(self.sess, ckpt_name)
        
    def resBlock(self,x,f_nr):
        y = tf.nn.conv2d(x, filter=self.resFilters[f_nr], strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.resBiases[f_nr]
        y = tf.nn.relu(x)
        y = tf.nn.conv2d(y, filter=self.resFilters[f_nr+1], strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.resBiases[f_nr+1]
        y = y*self.scaling_factor
        return x+y
    
    def GetWeight(self,shape,name = 'weight',std = 5e-2):
        init = tf.truncated_normal_initializer(stddev = std)
        return tf.get_variable(shape = shape,initializer = init,name = name)
    
    def srop(self,x):
        # -- Model architecture --

        # first conv
        y = tf.nn.conv2d(x, filter=self.filter_one, strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.bias_one
        y1 = tf.identity(y)
        
        # all residual blocks
        for i in range(self.B):
            y = self.resBlock(y,(i*2))
        
        # last conv
        y = tf.nn.conv2d(y, filter=self.filter_two, strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.bias_two
        y = y+y1
        
        # upsample via sub-pixel, equivalent to depth to space
        y = tf.nn.conv2d(y, filter=self.filter_three, strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.bias_three
        y = tf.nn.depth_to_space(y,self.scale,data_format = 'NHWC', name = 'NHWC_output')
        
        return y

    def Shrink(self,x,theta):
        return tf.multiply(tf.sign(x),tf.maximum(tf.abs(x)-theta,0))

    def Shrinkage(self,x,name):
        theta = self.GetWeight([1,256],name = name)
        alpha = tf.div(x,theta)
        alpha = self.Shrink(alpha,1)
        alpha = tf.multiply(alpha,theta)
        return alpha

    def RNConvISTA(self,Iy,H,scope = 'RNConvISTA',reuse = None):
        
        HT = tf.transpose(H)
        HTH = tf.matmul(HT,H)
        
        with tf.variable_scope(scope,reuse = reuse):
            _,im_h,im_w,_ = Iy.get_shape().as_list()
            sh = tf.shape(Iy)
            y = slim.conv2d(Iy,256,9,stride = 1,activation_fn = None,padding = 'SAME',scope = 'ConvH')
            y = tf.reshape(y,(-1,256))
            W = self.GetWeight([256,256],name = 'W')
            S = self.GetWeight([256,256],name = 'S')
            y = tf.matmul(y,HT)
            y = tf.matmul(y,W)
            alpha = y
            for i in range(2):
                alpha = self.Shrinkage(alpha,name = 'Theta{}'.format(i))
                alpha = tf.matmul(alpha,HTH)
                alpha = tf.matmul(alpha,S)
                alpha = y+alpha
            alpha = self.Shrinkage(alpha,name = 'ThetaEnd')
            Dx = self.GetWeight([256,25],name = 'Dx')
            x = tf.matmul(alpha,Dx)
            # x = tf.reshape(x,(-1,im_h,im_w,25,1))
            # Ix = tf.reduce_sum(x,3)
            #x = tf.reshape(x,(-1,im_h,im_w,25))
            nsh = tf.concat([sh[:3],tf.constant([25])],axis = 0)
            x = tf.reshape(x,nsh)
            Ix = slim.conv2d(x,3,5,stride = 1,activation_fn = None,padding = 'SAME',scope = 'ConvG')
            return Ix

    def slsr(self,x,H):
        I_est = self.srop(x)
        I_res = self.RNConvISTA(I_est, H)
        I_sr = I_res + I_est
        return I_sr
    
         
    def test(self,image):
        if image is None:
            raise AttributeError("U must input an image path or an image mtx.")
        elif type(image) == type(np.array([])):
            lr_image = image
        elif type(image) == type(''):
            from PIL import Image
            lr_image = Image.open(image)
            lr_image = np.array(lr_image)
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
        
        lr_image = lr_image[np.newaxis,:]
        blur_kernel = self.sess_ckp.run(self.ker_map, feed_dict={self.LR_tensor1: lr_image})
        blur_kernel = blur_kernel[0]
        iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        result = self.sess_slsr.run(self.HR_tensor, feed_dict={self.LR_tensor2: lr_image / 255.0, self.H:iH})
        x = np.squeeze(result) * 255.0
        x = np.clip(x, 0, 255)
        return x
        
    def eval(self,validfolder):
        
        data_set = os.path.split(validfolder)[0]
        data_set = os.path.split(data_set)[-1]
        
        # if not os.path.exists('./RESULT/SLKPRx{}/'.format(self.scale)+data_set):
        #     os.makedirs('./RESULT/SLKPSRx{}/'.format(self.scale)+data_set)
        
        print("\nPrepare Data...\n")
        paths = super().prepare_data(validfolder)
        data_num = len(paths)

        avg_time = 0
        avg_pasn = 0
        avg_ssim = 0
        print("\nNow Start Testing...\n")
        
        data_set = os.path.split(validfolder)[0]
        data_set = os.path.split(data_set)[-1]
        # print(data_set)
        # blur_kernel = fspecial(kernel_size = 17,sigma = 4)
        # iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        # pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(self.scale)
        # with gfile.FastGFile(pbPath,'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     self.sess.graph.as_default()
        #     tf.import_graph_def(graph_def,name = '')
        # LR_tensor = self.sess.graph.get_tensor_by_name("images:0")
        # HR_tensor = self.sess.graph.get_tensor_by_name("add_100:0")
        # H = self.sess.graph.get_tensor_by_name("H:0")
        ker = fspecial(kernel_size = 17,sigma = 3)
        
        for idx in range(data_num):
            input_ = cv.imread(paths[idx])
            label_ = np.array(input_)
            input_ = DegradeFilter(input_,blur_kernel = ker)
            input_ = input_[np.newaxis,:]
            # input_, label_ = super().get_image(paths[idx], self.scale, None)
            fn = os.path.split(paths[idx])[-1]
            fn = os.path.splitext(fn)[0]
            
            time_ = time()            
            blur_kernel = self.sess_ckp.run(self.ker_map, feed_dict={self.LR_tensor1: input_})
            blur_kernel = blur_kernel[0]
            iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
            
            result = self.sess_slsr.run(self.HR_tensor, feed_dict={self.LR_tensor2: input_, self.H:iH})
            
            avg_time += time() - time_

            x = np.squeeze(result)
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_, self.scale)
            avg_pasn += psnr
            issim = ssim(x, label_)
            avg_ssim += issim
            
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx+1, data_num, time() - time_ , psnr))

            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/SLKPSRx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/SLKPSRx{}/".format(self.scale)+data_set))
            #print( "./RESULT/SrOpx{}/".format(self.scale) + fn +"/_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            super().imsave(x, "./RESULT/SLKPSRx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            
        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)
