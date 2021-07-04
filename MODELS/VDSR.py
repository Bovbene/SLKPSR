# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:15:52 2021

@author: 月光下的云海
"""


from .BaseModel import BaseModel
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
from time import time,strftime,localtime
from glob import glob
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from DLL.utils import del_file,setup_logger,fspecial,DegradeFilter
import logging
from DLL.Valuation import PSNR,ssim

class VDSR(BaseModel):
    
    def __init__(self,scale,epoch):
        
        # -- Global Config --
        self.scale = scale
        self.epochs = epoch
        self.lr = 1e-4
        self.batch_size = 32
        self.save_path = './TRAINED_MODEL/'
        self.blk_size = 128
        self.lr_decay_rate = 0.5
        
        # -- GPU Config --
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
    def vdsr(self,x):
        lr_sh = tf.shape(x)
        y = tf.image.resize_bicubic(x, size = [lr_sh[1]*self.scale,lr_sh[2]*self.scale])
        with slim.arg_scope([slim.conv2d],activation_fn = super().lrelu):
            for i in range(14):
                y = slim.conv2d(y,64,3,stride = 1,padding = 'SAME',scope = 'conv{}'.format(i))
            y = slim.conv2d(y,3,3,stride = 1,padding = 'SAME',scope = 'conv15')
            y = y+x
        return y
    
    def train(self,trainfolder,validfolder):
        
        # Setup Logger
        setup_logger('base','./TRAINED_MODEL/','train_on_VDSRx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        ker = fspecial(kernel_size = 17,sigma = 3)
        
        # Create training dataset
        train_image_paths = glob(trainfolder)
        train_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_dataset,
                                                       output_types = (tf.float32,tf.float32),
                                                       output_shapes = (tf.TensorShape([None,None,3]),tf.TensorShape([None,None,3])),
                                                       args = [train_image_paths, self.scale, ker, self.blk_size] )
        train_dataset = train_dataset.padded_batch(self.batch_size, padded_shapes = ([None,None,3],[None,None,3]))
        
        # Create validation dataset
        val_image_paths = glob(validfolder)
        val_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_val_dataset,
                                                     output_types = (tf.float32,tf.float32),
                                                     output_shapes = (tf.TensorShape([None,None,3]),tf.TensorShape([None,None,3])),
                                                     args = [val_image_paths,self.scale,ker])
        val_dataset = val_dataset.padded_batch(1, padded_shapes = ([None,None,3],[None,None,3]))
        
        # Make the iterator and its initializer
        train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        train_initializer = train_val_iterator.make_initializer(train_dataset)
        val_initializer = train_val_iterator.make_initializer(val_dataset)
        
        handle = tf.placeholder(tf.string,shape = [])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        LR,HR = iterator.get_next()
        
        logger.info("\n Running SLSR")
        I_sr = self.vdsr(LR)
        print(I_sr)
        psnr = tf.image.psnr(I_sr,HR, max_val = 255.0)
        ssim = tf.image.ssim(I_sr,HR,max_val = 255.0)
        
        logger.info("\nThe parameters volumn is:")
        all_vars = tf.global_variables()
        logger.info(str(super().count_param(all_vars)))
        
        train_op,loss,lr = super().grenerate_train_op(HR,I_sr,self.batch_size*1e3,self.lr, lr_decay_rate = 0.5, var_list = None,loss_function = super().l1_loss)
        
        if os.path.exists('./LOGS/VDSRx{}'.format(self.scale)):
            del_file("./LOGS/VDSR_x{}/".format(self.scale))
        
        sess = tf.Session(config = self.config)
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./LOGS/VDSR_x{}'.format(self.scale), sess.graph)
        train_val_handle = sess.run(train_val_iterator.string_handle())
        
        logger.info("Training ... ...")
        flag = -float("Inf")
        for e in range(1,self.epochs+1):
            t = time()
            sess.run(train_initializer)
            step,train_loss = 0,0
            try:
                while True:
                    l,lr_,_ = sess.run([loss,lr,train_op],feed_dict = {handle:train_val_handle})
                    train_loss += l
                    step += 1
                    if step % 1000 == 0:
                        logger.info( "Step nr: [{}/{}] - Loss: {:.5f} - Lr: {:.6f}".format(step, "?", float(train_loss/step), lr_ ) )
            except tf.errors.OutOfRangeError:
                pass
            sess.run(val_initializer)
            tot_val_psnr,tot_val_ssim,val_im_cntr = 0, 0, 0
            try:
                while True:
                    val_psnr,val_ssim = sess.run([psnr,ssim],feed_dict = {handle:train_val_handle})
                    tot_val_psnr += val_psnr[0]
                    tot_val_ssim += val_ssim[0]
                    val_im_cntr += 1
            except tf.errors.OutOfRangeError:
                pass
            vpsnr,vssim = tot_val_psnr / val_im_cntr , tot_val_ssim / val_im_cntr
            logger.info("[{:.3f}]Epoch nr: [{}/{}]  - Loss: {:.5f} - val PSNR: {:.3f} - val SSIM: {:.3f}\n"
                        .format(time()-t,e,self.epochs,train_loss/step,vpsnr,vssim))
            if vpsnr > flag:
                logger.info("Saving the better model ... ...")
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["add"])
                with tf.gfile.FastGFile(self.save_path+'VDSR_x{}.pb'.format(self.scale), mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                flag = vpsnr
                ThSSIM = vssim
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}),(SSIM:{:.4f}).'.format(flag, ThSSIM))
        train_writer.close()
        sess.close()
        return "--<The training porcess of VDSR has been completed.>--"
    
    def eval(self, validfolder):
        print("\nPrepare Data...\n")
        paths = super().prepare_data(validfolder)
        data_num = len(paths)

        avg_time = 0
        avg_pasn = 0
        avg_ssim = 0
        print("\nNow Start Testing...\n")
        
        ker = fspecial(kernel_size = 17,sigma = 3)
        
        data_set = os.path.split(validfolder)[0]
        data_set = os.path.split(data_set)[-1]
        
        sess = tf.Session(config = self.config)
        
        pbPath = "./TRAINED_MODEL/VDSR_x{}.pb".format(self.scale)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        
        images = sess.graph.get_tensor_by_name("IteratorGetNext:0")
        pred = sess.graph.get_tensor_by_name("Relu_2:0")
        
        for idx in range(data_num):
            input_ = cv.imread(paths[idx])
            label_ = np.array(input_)
            input_ = DegradeFilter(input_,blur_kernel = ker)
            input_ = input_[np.newaxis,:]
            
            fn = os.path.split(paths[idx])[-1]
            fn = os.path.splitext(fn)[0]
            
            time_ = time()
            result = sess.run([pred], feed_dict={images: input_})
            avg_time += time() - time_

            x = np.squeeze(result)
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_, self.scale)
            avg_pasn += psnr
            issim = ssim(x, label_)
            avg_ssim += issim
            
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx+1, data_num, time() - time_ , psnr))

            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/VDSRx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/VDSRx{}/".format(self.scale)+data_set))
            #print( "./RESULT/SrOpx{}/".format(self.scale) + fn +"/_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            super().imsave(x, "./RESULT/VDSRx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )
        sess.close()
        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)
                    
        
        
        
        
        
        
        

