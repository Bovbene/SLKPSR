# -*- coding: utf-8 -*-
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
from DLL.utils import fspecial,GetBlurMtx,setup_logger,del_file
import tensorflow.contrib.slim as slim
from DLL.Valuation import ssim,PSNR


class SLSR(BaseModel):
    
    def __init__(self,scale,epoch):
        
        # -- Global Config --
        self.scale = scale
        self.epochs = epoch
        self.lr = 1e-4
        self.ch = 3
        
        # -- GPU Config --
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        # -- SrOp config --
        self.B = 32 #Number of resBlocks
        self.F = 256 #Number of filters
        self.batch_size = 48
        self.scaling_factor = 0.1
        self.PS = 3 * (self.scale*self.scale) #channels x scale^2
        self.is_norm = False
        if self.is_norm:
            self.mean = [103.1545782, 111.561547, 114.35629928]
        else:
            self.mean = [0,0,0]
        self.save_path = './TRAINED_MODEL/'
        
        tf.reset_default_graph()
        self.sess = tf.Session(config=self.config)
        self.xavier = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.constant_initializer(value=0.0)
        # self.build_rnconvista()
        self.blk_size = 64
        self.stride = 16
        self.c_dim = 3
        self.lr_decay_steps = 5
        self.lr_decay_rate = 0.5
        self.build_srop()
        
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
        if self.scale == 1: return y
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
        
        #l,lr_h,lr_w,ch = Iy.get_shape().as_list()
        #Iy = tf.image.resize_bicubic(Iy, (lr_h*self.scale_factor,lr_w*self.scale_factor))
        #Iy = slim.conv2d_transpose(Iy, ch, 5,stride = self.scale,padding = 'SAME',scope = 'Deconv')
        
        HT = tf.transpose(H)
        HTH = tf.matmul(HT,H)
        
        # H = GetBlurMtx(blur_kernel,1,imshape = (16,16))
        # HT = tf.cast(tf.constant(H.T.toarray()),dtype = tf.float32)
        # HTH = tf.cast(tf.constant((H.T @ H).toarray()),dtype = tf.float32)
        
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
    
    def initialize_uninitialized(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))
    
    def train(self,imagefolder,validfolder):
        if self.scale == 1: return self.train_blurimg(imagefolder,validfolder)
        else: return self.train_scale(imagefolder,validfolder)
    
    def train_scale(self,imagefolder,validfolder):
        #Setup Logger and Blur Kernel
        setup_logger('base','./TRAINED_MODEL/','train_on_SLSRx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        blur_kernel = fspecial(kernel_size = 17,sigma = 4)
        iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        
        # Create training dataset
        data = super().input_setup(imagefolder,self.scale,self.blk_size,self.stride)
        if len(data) == 0:
            logger.info("\nCan Not Find Training Data!\n")
            return
        
        data_dir = super().get_data_dir("./TRAINED_MODEL/", self.scale)
        data_num = super().get_data_num(data_dir)
        batch_num = data_num // self.batch_size
        self.images = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='images')
        H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
        self.labels = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='labels')
        self.pred = self.slsr(self.images,H)
        counter = 0
        epoch_start = counter // batch_num
        batch_start = counter % batch_num
        
        logger.info("\nThe parameters volumn is:")
        all_vars = tf.global_variables()
        logger.info(super().count_param(all_vars))
        
        res_vars = [v for v in all_vars if ('RNConvISTA' in v.name)]
        
        learning_step,self.loss,learning_rate = super().grenerate_train_op(self.pred, 
                                                                           self.labels, 
                                                                           self.lr_decay_steps*batch_num, 
                                                                           self.lr, 
                                                                           self.lr_decay_rate, 
                                                                           var_list = res_vars,
                                                                           loss_function = super().l1_loss)
        
        self.summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver(max_to_keep=10)
        
        self.initialize_uninitialized()

        merged_summary_op = tf.summary.merge_all()
        if os.path.exists('./LOGS/SLSR'):
            del_file("./LOGS/SLSR/")
        summary_writer = tf.summary.FileWriter('./LOGS/SLSR', self.sess.graph)
        
        test_path = super().prepare_data(validfolder)
        
        flag = -float("Inf")
        logger.info("Start Training SLSR ... ...")
        for ep in range(epoch_start, self.epochs):
            # Run by batch images
            for idx in range(batch_start, batch_num):
                batch_images, batch_labels = super().get_batch(data_dir, data_num, self.batch_size)
                counter += 1
                _, err, lr = self.sess.run([learning_step, self.loss, learning_rate], feed_dict={self.images: batch_images, H:iH, self.labels: batch_labels})
                if counter % 10 == 0:
                    logger.info("Epoch: [%4d], batch: [%6d/%6d], loss: [%.8f], lr: [%.6f], step: [%d]" % ((ep+1), (idx+1), batch_num, err, lr, counter))
                if counter % 1000 == 0:
                    avg_psnr = 0
                    for p in test_path:
                        input_, label_ = super().get_image(p, self.scale, None)
                        sr = self.sess.run(self.pred, feed_dict={self.images: input_ / 255.0, H:iH})
                        sr = np.squeeze(sr) * 255.0
                        sr = np.clip(sr, 0, 255)
                        psnr = PSNR(sr, label_[0], self.scale)
                        avg_psnr += psnr
                    logger.info("Ave PSNR is:" + str(avg_psnr/len(test_path)) )
                    if avg_psnr>flag:
                        logger.info("Saving the better model... ...")
                        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["add_100"])
                        with tf.gfile.FastGFile(self.save_path+'SLSR_x{}.pb'.format(self.scale), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        flag = avg_psnr
                    summary_str = self.sess.run(merged_summary_op, feed_dict={self.images: batch_images, H:iH, self.labels: batch_labels})
                    summary_writer.add_summary(summary_str, counter)
                    
                if counter > 0 and counter == batch_num * self.epochs:
                    avg_psnr = 0
                    for p in test_path:
                        input_, label_ = super().get_image(p, self.scale, None)
                        sr = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
                        sr = np.squeeze(sr) * 255.0
                        sr = np.clip(sr, 0, 255)
                        psnr = PSNR(sr, label_[0], self.scale)
                        avg_psnr += psnr
                    logger.info("Ave PSNR is:" + str(avg_psnr/len(test_path)) )
                    if avg_psnr>flag:
                        logger.info("Saving the better model... ...")
                        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["add_100"])
                        with tf.gfile.FastGFile(self.save_path+'SLSR_x{}.pb'.format(self.scale), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        flag = avg_psnr
                    break

        summary_writer.close()
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}).'.format(flag))
        logger.info("--<The training porcess of SrOp has been completed.>--")
        return "--<The training porcess of SrOp has been completed.>--"
        
    def train_blurimg(self,trainfolder,validfolder):
        
        #Setup Logger and Blur Kernel
        setup_logger('base','./TRAINED_MODEL/','train_on_SLSRx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        ker = fspecial(kernel_size = 17,sigma = 3)
        iH = GetBlurMtx(ker,1,imshape = (16,16)).toarray().astype(np.float32)

        # Create training dataset
        train_image_paths = glob(trainfolder+'*.png')
        # train_dataset = tf.data.Dataset.from_generator(generator=super().make_dataset,
        #                                          output_types=(tf.float32, tf.float32),
        #                                          output_shapes=(tf.TensorShape([None, None, self.ch]), tf.TensorShape([None, None, self.ch])),
        #                                          args=[train_image_paths, self.scale, self.mean,self.blk_size])
        train_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, self.ch]), tf.TensorShape([None, None, self.ch])),
                                                 args=[train_image_paths, self.scale, ker, self.blk_size])
        train_dataset = train_dataset.padded_batch(self.batch_size, padded_shapes=([None, None, self.ch], [None, None, self.ch]))
        
        # Create validation dataset
        val_image_paths = glob(validfolder+'*.bmp')
        # val_dataset = tf.data.Dataset.from_generator(generator=super().make_val_dataset,
        #                                          output_types=(tf.float32, tf.float32),
        #                                          output_shapes=(tf.TensorShape([None, None, self.ch]), tf.TensorShape([None, None, self.ch])),
        #                                          args=[val_image_paths, self.scale, self.mean])
        val_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_val_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, self.ch]), tf.TensorShape([None, None, self.ch])),
                                                 args=[val_image_paths, self.scale, ker])
        val_dataset = val_dataset.padded_batch(1, padded_shapes=([None, None, self.ch],[None, None, self.ch]))

        # Make the iterator and its initializers
        train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = train_val_iterator.make_initializer(train_dataset)
        val_initializer = train_val_iterator.make_initializer(val_dataset)
        
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        LR, HR = iterator.get_next()
        H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
        
        logger.info("\nRunning SLSR... ...")
        I_sr = self.slsr(LR,iH)
        psnr = tf.image.psnr(I_sr, HR, max_val=255.0)
        ssim = tf.image.ssim(I_sr, HR, max_val=255.0)
        print("\nThe parameters volumn is:")
        all_vars = tf.global_variables()
        print(super().count_param(all_vars))
        # est_vars = [v for v in all_vars if ('SrOp' in v.name)]
        res_vars = [v for v in all_vars if ('RNConvISTA' in v.name)]
        # train_op,loss,lr = super().grenerate_train_op(HR, I_sr, self.batch_size*1e3, self.lr, lr_decay_rate = self.lr_decay_rate, var_list = res_vars,loss_function = super().l1_loss)
        
        loss = super().l1_loss(HR,I_sr)
        global_step = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(self.lr,
                                        global_step,
                                        decay_steps=15000,
                                        decay_rate=0.95,
                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss,var_list = res_vars))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        
        if os.path.exists('./LOGS/SLSR_x{}'.format(self.scale)):
            del_file("./LOGS/SLSR_x{}/".format(self.scale))
        train_writer = tf.summary.FileWriter('./LOGS/SLSR_x{}'.format(self.scale), self.sess.graph)
        train_val_handle = self.sess.run(train_val_iterator.string_handle())
                
        logger.info("Training... ...")
        flag = -float('Inf')
        for e in range(1,self.epochs+1):
            start_t = time()
            self.sess.run(train_initializer)
            #self.sess.run(tf.global_variables_initializer())
            self.initialize_uninitialized()
            step, train_loss = 0, 0
            try:
                while True:
                    o,l,_ = self.sess.run([I_sr,loss,train_op],feed_dict = {handle:train_val_handle,H:iH})
                    train_loss += l
                    step += 1
                    if step % 1000 == 0:
                        logger.info("Step nr: [{}/{}] - Loss: {:.5f}".format(step, "?", float(train_loss/step) ))
            except tf.errors.OutOfRangeError:
                pass
            self.sess.run(val_initializer)
            tot_val_psnr, tot_val_ssim, val_im_cntr = 0, 0, 0
            try:
                while True:
                    val_psnr, val_ssim = self.sess.run([psnr, ssim], feed_dict={handle:train_val_handle,H:iH})
                    tot_val_psnr += val_psnr[0]
                    tot_val_ssim += val_ssim[0]
                    val_im_cntr += 1
            except tf.errors.OutOfRangeError:
                pass
            vpsnr,vssim = tot_val_psnr / val_im_cntr , tot_val_ssim / val_im_cntr
            logger.info("[{:.3f}]Epoch nr: [{}/{}]  - Loss: {:.5f} - val PSNR: {:.3f} - val SSIM: {:.3f}\n"
                  .format(time()-start_t,e,self.epochs,train_loss/step,vpsnr,vssim))
            if vpsnr > flag:
                logger.info('Saving the better model ... ...')
                constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["add_100"])
                with tf.gfile.FastGFile(self.save_path+'SLSR_x{}.pb'.format(self.scale), mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                flag = vpsnr
                ThSSIM = vssim
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}),(SSIM:{:.4f}).'.format(flag, ThSSIM))
        train_writer.close()
        self.sess.close()
        
    def test2(self,image,blur_kernel = None):
        if image is None:
            raise AttributeError("U must input an image path or an image mtx.")
        elif type(image) == type(np.array([])):
            lr_image = image
        elif type(image) == type(''):
            lr_image = cv.imread(image)
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
        if blur_kernel is None:
            blur_kernel = fspecial(kernel_size = 17,sigma = 4)
        lr_image = image.astype(np.float32) - self.mean
        lr_image = lr_image.reshape((1,)+lr_image.shape)
        iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(self.scale)
        sess = tf.Session(config = self.config)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = sess.graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = sess.graph.get_tensor_by_name("add_100:0")
        # H = sess.graph.get_tensor_by_name("H:0")
        H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
        res = sess.run(HR_tensor, feed_dict={LR_tensor: lr_image,H:iH})
        sess.close()
        res = res[0]
        res = (res + self.mean).clip(min = 0, max = 255)
        res = res.astype(np.uint8)
        return res
            
    def test(self,image,blur_kernel = None):
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
        if blur_kernel is None:
            blur_kernel = fspecial(kernel_size = 17,sigma = 4)
        input_ = lr_image[np.newaxis,:]/255.0
        iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        print(iH.shape)
        pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(self.scale)
        #sess = tf.Session(config = self.config)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = self.sess.graph.get_tensor_by_name("images:0")
        # LR_tensor = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='images')
        HR_tensor = self.sess.graph.get_tensor_by_name("add_100:0")
        # H = tf.placeholder(shape = [256,256],dtype = tf.float32,name = 'H')
        H = self.sess.graph.get_tensor_by_name("H:0")
        time_ = time()
        
        res = self.sess.run(HR_tensor, feed_dict={LR_tensor: input_,H:iH})
        x = np.squeeze(res) * 255.0
        x = np.clip(x, 0, 255)
        print("Time Elapsed:", time()-time_)
        return np.uint8(x)
        
    def eval(self,validfolder):
        
        data_set = os.path.split(validfolder)[0]
        data_set = os.path.split(data_set)[-1]
        
        if not os.path.exists('./RESULT/SLSRx{}/'.format(self.scale)+data_set):
            os.makedirs('./RESULT/SLSRx{}/'.format(self.scale)+data_set)
        
        print("\nPrepare Data...\n")
        paths = super().prepare_data(validfolder)
        data_num = len(paths)

        avg_time = 0
        avg_pasn = 0
        avg_ssim = 0
        print("\nNow Start Testing...\n")
        
        blur_kernel = fspecial(kernel_size = 17,sigma = 4)
        iH = GetBlurMtx(blur_kernel,1,imshape = (16,16)).toarray().astype(np.float32)
        pbPath = "./TRAINED_MODEL/SLSR_x{}.pb".format(self.scale)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = self.sess.graph.get_tensor_by_name("images:0")
        HR_tensor = self.sess.graph.get_tensor_by_name("add_100:0")
        H = self.sess.graph.get_tensor_by_name("H:0")
        
        for idx in range(data_num):
            input_, label_ = super().get_image(paths[idx], self.scale, None)
            fn = os.path.split(paths[idx])[-1]
            fn = os.path.splitext(fn)[0]
            
            time_ = time()
            result = self.sess.run([HR_tensor], feed_dict={LR_tensor: input_ / 255.0 , H:iH})
            
            avg_time += time() - time_

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_[0], self.scale)
            avg_pasn += psnr
            issim = ssim(x, label_[0])
            avg_ssim += issim
            
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time() - time_ , psnr))

            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/SLSRx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/SLSRx{}/".format(self.scale)+data_set))
            #print( "./RESULT/SrOpx{}/".format(self.scale) + fn +"/_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            super().imsave(x[:, :, ::-1], "./RESULT/SLSRx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            
        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)
