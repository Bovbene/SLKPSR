# -*- coding: utf-8 -*-
"""===============================================================================================
The Python code of SrOp
---------------------------------------------------------------------------------------------------
Class: SrOp
Param: 	self.B = 32 #Number of resBlocks
        self.F = 256 #Number of filters
        self.batch_size = 16 
        self.epochs = 1
        self.scale = scale
        self.lr = 1e-4
        self.scaling_factor = 0.1
        self.PS = 3 * (self.scale*self.scale) #channels x scale^2
        self.is_norm = True
        if self.is_norm:
            self.mean = [103.1545782, 111.561547, 114.35629928]
        else:
            self.mean = [0,0,0]
        self.save_path = './TRAINED_MODEL/'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.blk_size = 32
---------------------------------------------------------------------------------------------------
Tip: None
---------------------------------------------------------------------------------------------------
Created on Sat May  1 12:21:44 2021
@author: 月光下的云海(西电博巍)
Version: Ultimate
==============================================================================================="""

from .BaseModel import BaseModel
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
from time import time,strftime,localtime
from glob import glob
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from DLL.utils import del_file,setup_logger,fspecial,DegradeFilter
import logging
from DLL.Valuation import ssim,PSNR


class SrOp(BaseModel):
    
    def __init__(self,scale = 4,epoch = 10):
        
        # --Basic Settings--
        self.B = 32 #Number of resBlocks
        self.F = 256 #Number of filters
        self.batch_size = 48 
        self.epochs = epoch
        self.scale = scale
        self.lr = 1e-4
        self.scaling_factor = 0.1
        self.PS = 3 * (self.scale*self.scale) #channels x scale^2
        self.is_norm = False
        if self.is_norm:
            self.mean = [103.1545782, 111.561547, 114.35629928]
        else:
            self.mean = [0,0,0]
        self.save_path = './TRAINED_MODEL/'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.blk_size = 64
        self.stride = 16
        self.c_dim = 3
        self.lr_decay_steps = 10
        self.lr_decay_rate = 0.5
        self.build()
        
    def build(self):
        tf.reset_default_graph()
        self.xavier = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.constant_initializer(value=0.0)
        
        # self.xavier = tf.truncated_normal_initializer(stddev = 5e-2)
        # self.bias_initializer = tf.constant_initializer(0.0)
             
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
        
    def resBlock(self,x,f_nr):
        y = tf.nn.conv2d(x, filter=self.resFilters[f_nr], strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.resBiases[f_nr]
        y = tf.nn.relu(x)
        y = tf.nn.conv2d(y, filter=self.resFilters[f_nr+1], strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.resBiases[f_nr+1]
        y = y*self.scaling_factor
        out = x+y
        return out
    
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
    
    def train(self,imagefolder,validfolder):
        if self.scale == 1: return self.train_blurimg(imagefolder,validfolder)
        else: return self.train_scale(imagefolder,validfolder)
    
    def train_blurimg(self,imagefolder,validfolder):
        
        setup_logger('base','./TRAINED_MODEL/','train_on_SrOpx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        
        ker = fspecial(kernel_size = 17,sigma = 3)
        
        # Create training dataset
        train_image_paths = glob(imagefolder+'*.png')
        # train_dataset = tf.data.Dataset.from_generator(generator = super().make_sub_data,
        #                                           output_types=(tf.float32, tf.float32),
        #                                           output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
        #                                           args=[train_image_paths, self.scale, self.blk_size, self.stride])
        train_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_dataset,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                  args=[train_image_paths, self.scale, ker, self.blk_size])
        train_dataset = train_dataset.padded_batch(self.batch_size, padded_shapes=([None, None, 3],[None, None, 3]))
        
        # Create validation dataset
        val_image_paths = glob(validfolder+'*.bmp')
        # val_dataset = tf.data.Dataset.from_generator(generator = super().make_val_dataset,
        #                                           output_types=(tf.float32, tf.float32),
        #                                           output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
        #                                           args=[val_image_paths, self.scale])
        val_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_val_dataset,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                  args=[val_image_paths, self.scale, ker])
        val_dataset = val_dataset.padded_batch(1, padded_shapes=([None, None, 3],[None, None, 3]))
        
        # Make the iterator and its initializers
        train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = train_val_iterator.make_initializer(train_dataset)
        val_initializer = train_val_iterator.make_initializer(val_dataset)
        
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        LR, HR = iterator.get_next()
        
        # SrOp model
        logger.info("\nRunning SrOp... ...")
        out = self.srop(x = LR)
        psnr = tf.image.psnr(out, HR, max_val=255.0)
        ssim = tf.image.ssim(out, HR, max_val=255.0)
        train_op,loss,lr = super().grenerate_train_op(HR,out,self.batch_size*1e3,self.lr, lr_decay_rate = 0.9, var_list = None,loss_function = super().l1_loss)
        # loss = super().l1_loss(HR,out)
        # train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        
        # loss = super().l1_loss(HR,out)
        # global_step = tf.Variable(0,trainable = False)
        # lr = tf.train.exponential_decay(self.lr,
        #                                 global_step,
        #                                 decay_steps=20000,
        #                                 decay_rate=0.5,
        #                                 staircase=True)
        # optimizer = tf.train.AdamOptimizer(lr)
        # gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # train_op = optimizer.apply_gradients(zip(gradients, variables))
        
        
        logger.info( "\nThe parameters volumn is:" + str( super().count_param(tf.global_variables()) ) )
        
        if os.path.exists('./LOGS/SrOp'):
            del_file("./LOGS/SrOp/")
            
        sess =  tf.Session(config=self.config)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('./LOGS/SrOp', sess.graph)
        sess.run(tf.global_variables_initializer())
        train_val_handle = sess.run(train_val_iterator.string_handle())
        logger.info("Training SrOp ... ...")
        flag = -float('Inf')
        
        for e in range(1,self.epochs+1):
            sess.run(train_initializer)
            step, train_loss = 0, 0
            try:
                while True:
                    lr_,o,l,_ = sess.run([lr,out,loss,train_op],feed_dict = {handle:train_val_handle})
                    train_loss += l
                    step += 1
                    if step % 1000 == 0:
                        logger.info( "Step nr: [{}/{}] - Loss: {:.5f} - lr:{:.5f}".format(step, "?", float(train_loss/step), lr_ ) )
            except tf.errors.OutOfRangeError:
                pass
            sess.run(val_initializer)
            tot_val_psnr, tot_val_ssim, val_im_cntr = 0, 0, 0
            start_t = time()
            try:
                while True:
                    val_psnr, val_ssim = sess.run([psnr, ssim], feed_dict={handle:train_val_handle})
                    tot_val_psnr += val_psnr[0]
                    tot_val_ssim += val_ssim[0]
                    val_im_cntr += 1
            except tf.errors.OutOfRangeError:
                pass
            train_loss,vpsnr,vssim, = (train_loss/step),(tot_val_psnr / val_im_cntr),(tot_val_ssim / val_im_cntr)
            logger.info( "[{:.3f}] Epoch nr: [{}/{}]  - Loss: {:.5f} - val PSNR: {:.3f} - val SSIM: {:.3f}\n"
                  .format(time()-start_t,e,self.epochs,train_loss,vpsnr,vssim) )
            if vpsnr > flag:
                logger.info('Saving the better model ... ...')
                saver.save(sess, self.save_path + "SrOp_x{}/model.ckpt".format(self.scale))
                flag = vpsnr
        train_writer.close()
        sess.close()
        return "--<The training porcess of SrOp has been completed.>--"
    
    def test2(self,image):
        if image is None:
            raise AttributeError("U must input an image path or an image mtx.")
        elif type(image) == type(np.array([])):
            lr_image = image
        elif type(image) == type(''):
            lr_image = cv.imread(image)
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
        lr_image = image.astype(np.float32) - self.mean
        lr_image = lr_image.reshape((1,)+lr_image.shape)
        pbPath = "./TRAINED_MODEL/SrOp_x{}.pb".format(self.scale)
        sess = tf.Session()
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = sess.graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = sess.graph.get_tensor_by_name("NHWC_output:0")
        res = sess.run(HR_tensor, feed_dict={LR_tensor: lr_image})
        sess.close()
        res = res[0]
        res = (res + self.mean).clip(min = 0, max = 255)
        res = res.astype(np.uint8)
        return res
    
    
    def train_scale(self, imagefolder,validfolder):
        
        setup_logger('base','./TRAINED_MODEL/','train_on_SrOpx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        
        logger.info("Prepare Data...")
        data = super().input_setup(imagefolder,self.scale,self.blk_size,self.stride)
        if len(data) == 0:
            logger.info("\nCan Not Find Training Data!\n")
            return

        data_dir = super().get_data_dir("./TRAINED_MODEL/", self.scale)
        data_num = super().get_data_num(data_dir)
        batch_num = data_num // self.batch_size

        # images_shape = [None, self.blk_size, self.blk_size, self.c_dim]
        # labels_shape = [None, self.blk_size * self.scale, self.blk_size * self.scale, self.c_dim]
        
        self.images = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='labels')
        self.pred = self.srop(self.images)
        #self.loss = super().l1_loss(self.labels,self.pred)
        

        counter = 0
        epoch_start = counter // batch_num
        batch_start = counter % batch_num

        logger.info("The parameters volumn is:")
        all_vars = tf.global_variables()
        logger.info(super().count_param(all_vars))
        
        learning_step,self.loss,learning_rate = super().grenerate_train_op(self.pred, 
                                                                           self.labels, 
                                                                           self.lr_decay_steps*batch_num, 
                                                                           self.lr, 
                                                                           self.lr_decay_rate, 
                                                                           var_list = None,
                                                                           loss_function = super().l1_loss)
        
        self.summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session(config = self.config)
        
        # global_step = tf.Variable(counter, trainable=False)
        # learning_rate = tf.train.exponential_decay(self.lr, global_step, self.lr_decay_steps*batch_num, self.lr_decay_rate, staircase=True)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # learning_step = optimizer.minimize(self.loss, global_step=global_step)

        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        if os.path.exists('./LOGS/SrOp'):
            del_file("./LOGS/SrOp/")
        summary_writer = tf.summary.FileWriter('./LOGS/SrOp', self.sess.graph)

        test_path = super().prepare_data(validfolder)
        
        flag = -float("Inf")
        logger.info("Now Start Training SrOp ... ...")
        for ep in range(epoch_start, self.epochs):
            # Run by batch images
            for idx in range(batch_start, batch_num):
                batch_images, batch_labels = super().get_batch(data_dir, data_num, self.batch_size)
                counter += 1

                _, err, lr = self.sess.run([learning_step, self.loss, learning_rate], feed_dict={self.images: batch_images, self.labels: batch_labels})

                if counter % 10 == 0:
                    logger.info("Epoch: [%4d], batch: [%6d/%6d], loss: [%.8f], lr: [%.6f], step: [%d]" % ((ep+1), (idx+1), batch_num, err, lr, counter))
                if counter % 10000 == 0:
                    avg_psnr = 0
                    for p in test_path:
                        input_, label_ = super().get_image(p, self.scale, None)
                        sr = self.sess.run(self.pred, feed_dict={self.images: input_ / 255.0})
                        sr = np.squeeze(sr) * 255.0
                        sr = np.clip(sr, 0, 255)
                        psnr = PSNR(sr, label_[0], self.scale)
                        avg_psnr += psnr
                    logger.info("Ave PSNR is:" + str(avg_psnr/len(test_path)) )
                    if avg_psnr>flag:
                        logger.info("Saving the better model... ...")
                        self.saver.save(self.sess, self.save_path + "SrOp_x4/model.ckpt")
                        flag = avg_psnr
                    summary_str = self.sess.run(merged_summary_op, feed_dict={self.images: batch_images, self.labels: batch_labels})
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
                        self.saver.save(self.sess, self.save_path + "SrOp_x4/model.ckpt")
                        flag = avg_psnr
                    break

        summary_writer.close()
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}).'.format(flag))
        logger.info("--<The training porcess of SrOp has been completed.>--")
        return "--<The training porcess of SrOp has been completed.>--"
    
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
            
        input_ = lr_image[np.newaxis,:]
        self.images = tf.placeholder(tf.float32, shape = [None,None,None,3], name='images')
        self.labels = tf.placeholder(tf.float32, shape = [None,None,None,3], name='labels')
        self.pred = self.srop(self.images)
        self.sess = tf.Session(config = self.config)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,self.save_path + "SrOp_x4/model.ckpt")
        #self.load("./TRAINED_MODEL/", restore=True)
        time_ = time()
        result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
        x = np.squeeze(result) * 255.0
        x = np.clip(x, 0, 255)
        print("Time Elapsed:", time()-time_)
        return np.uint8(x)
    
    def eval(self, validfolder):
        print("\nPrepare Data...\n")
        paths = super().prepare_data(validfolder)
        data_num = len(paths)

        avg_time = 0
        avg_pasn = 0
        avg_ssim = 0
        print("\nNow Start Testing...\n")
        
        data_set = os.path.split(validfolder)[0]
        data_set = os.path.split(data_set)[-1]
        
        self.sess = tf.Session(config = self.config)
        self.images = tf.placeholder(tf.float32, shape = [None,None,None,3], name='images')
        self.labels = tf.placeholder(tf.float32, shape = [None,None,None,3], name='labels')
        self.pred = self.srop(self.images)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,self.save_path + "SrOp_x{}/model.ckpt".format(self.scale))
        for idx in range(data_num):
            input_, label_ = super().get_image(paths[idx], self.scale, None)
            fn = os.path.split(paths[idx])[-1]
            fn = os.path.splitext(fn)[0]
            
            time_ = time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time() - time_

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_[0], self.scale)
            avg_pasn += psnr
            issim = ssim(x, label_[0])
            avg_ssim += issim
            
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time() - time_ , psnr))

            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/SrOpx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/SrOpx{}/".format(self.scale)+data_set))
            #print( "./RESULT/SrOpx{}/".format(self.scale) + fn +"/_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            super().imsave(x[:, :, ::-1], "./RESULT/SrOpx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )

        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)
        
    def eval2(self, validfolder):
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
        
        self.sess = tf.Session(config = self.config)
        self.images = tf.placeholder(tf.float32, shape = [None,None,None,3], name='images')
        self.labels = tf.placeholder(tf.float32, shape = [None,None,None,3], name='labels')
        self.pred = self.srop(self.images)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,self.save_path + "SrOp_x{}/model.ckpt".format(self.scale))
        for idx in range(data_num):
            input_ = cv.imread(paths[idx])
            label_ = np.array(input_)
            input_ = DegradeFilter(input_,blur_kernel = ker)
            input_ = input_[np.newaxis,:]
            
            fn = os.path.split(paths[idx])[-1]
            fn = os.path.splitext(fn)[0]
            
            time_ = time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_})
            avg_time += time() - time_

            x = np.squeeze(result)
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_, self.scale)
            avg_pasn += psnr
            issim = ssim(x, label_)
            avg_ssim += issim
            
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx+1, data_num, time() - time_ , psnr))

            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/SrOpx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/SrOpx{}/".format(self.scale)+data_set))
            #print( "./RESULT/SrOpx{}/".format(self.scale) + fn +"/_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            super().imsave(x, "./RESULT/SrOpx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )

        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)
       

