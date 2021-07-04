# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:33:50 2021

@author: 月光下的云海
"""

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
from DLL.Valuation import PSNR,ssim

class SRCNN(BaseModel):
    
    def __init__(self,scale,epoch):
        self.scale = scale
        self.lr = 1e-4
        self.batch_size = 32
        self.ch = 3
        self.epochs = epoch
        self.save_path = './TRAINED_MODEL/'
        self.blk_size = 32
        self.mean = [0,0,0]
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.blk_size = 128
        self.stride = 16
        self.c_dim = 3
        self.lr_decay_steps = 10
        self.lr_decay_rate = 0.5
        self.build()
        
    def build(self):
        tf.reset_default_graph()
        self.filter_one = super().kernel_init(shape = [3,3,self.ch,64],name = "SRCNN/Filter_one")
        self.filter_two = super().kernel_init(shape = [1,1,64,32],name = "SRCNN/Filter_two")
        self.filter_three = super().kernel_init(shape = [5,5,32,self.ch],name = "SRCNN/Filter_three")
        
        self.bias_one = super().bias_init(shape = [64],name = "SRCNN/Bias1")
        self.bias_two = super().kernel_init(shape = [32],name = "SRCNN/Bias2")
        self.bias_three = super().kernel_init(shape = [self.ch],name = "SRCNN/Bias3")
        
    def srcnn(self,x):
        lr_sh = tf.shape(x)
        y = tf.image.resize_bicubic(x, size = [lr_sh[1]*self.scale,lr_sh[2]*self.scale])
        y = super().conv2d(y,self.filter_one,self.bias_one,strides = 1,padding = "SAME",act = tf.nn.relu)
        y = super().conv2d(y,self.filter_two,self.bias_two,strides = 1,padding = "SAME",act = tf.nn.relu)
        y = super().conv2d(y,self.filter_three,self.bias_three,strides = 1,padding = "SAME",act = tf.nn.relu)
        return y
    
    def train(self,imagefolder,validfolder):
        if self.scale == 1: return self.train_blurimg(imagefolder,validfolder)
        else: return self.train_scale(imagefolder,validfolder)
    
    def train_blurimg(self,imagefolder,validfolder):
        
        setup_logger('base','./TRAINED_MODEL/','train_on_SRCNNx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        
        ker = fspecial(kernel_size = 17,sigma = 3)
        
        train_image_paths = glob(imagefolder)
        train_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_dataset,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                  args=[train_image_paths, self.scale, ker, self.blk_size])
        train_dataset = train_dataset.padded_batch(self.batch_size, padded_shapes=([None, None, 3],[None, None, 3]))
        
        # Create validation dataset
        val_image_paths = glob(validfolder)
        val_dataset = tf.data.Dataset.from_generator(generator = super().make_blur_val_dataset,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                  args=[val_image_paths, self.scale, ker])
        val_dataset = val_dataset.padded_batch(1, padded_shapes=([None, None, 3],[None, None, 3]))
        
        train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        train_initializer = train_val_iterator.make_initializer(train_dataset)
        val_initializer = train_val_iterator.make_initializer(val_dataset)
        
        handle = tf.placeholder(tf.string,shape = [])
        iterator = tf.data.Iterator.from_string_handle(handle,train_dataset.output_types,train_dataset.output_shapes)
        LR,HR = iterator.get_next()
        
        logger.info("Training SRCNN ... ...")
        out = self.srcnn(x = LR)
        print(out)
        psnr = tf.image.psnr(out,HR,max_val = 255)
        ssim = tf.image.ssim(out,HR,max_val = 255)
        
        train_op,loss,lr = super().grenerate_train_op(HR,out,self.batch_size*1e3,self.lr, lr_decay_rate = 0.5, var_list = None,loss_function = super().l1_loss)
        
        logger.info("\nThe parameters volumn is:")
        logger.info(super().count_param(tf.global_variables()))
        
        if os.path.exists('./LOGS/SRCNNx{}'.format(self.scale)):
            del_file("./LOGS/SRCNNx{}/".format(self.scale))
        
        sess = tf.Session(config = self.config)
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./LOGS/SRCNN', sess.graph)
        train_val_handle = sess.run(train_val_iterator.string_handle())
        
        logger.info("Training ... ...")
        flag = -float("Inf")
        for e in range(1,self.epochs+1):
            sess.run(train_initializer)
            start_t = time()
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
                        .format(time()-start_t,e,self.epochs,train_loss/step,vpsnr,vssim))
            if vpsnr > flag:
                logger.info("Saving the better model ... ...")
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Relu_2"])
                with tf.gfile.FastGFile(self.save_path+'SRCNN_x{}.pb'.format(self.scale), mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                flag = vpsnr
                ThSSIM = vssim
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}),(SSIM:{:.4f}).'.format(flag, ThSSIM))
        train_writer.close()
        sess.close()
        return "--<The training porcess of SRCNN has been completed.>--"
    
    def train_scale(self,imagefolder,validfolder):
        setup_logger('base','./TRAINED_MODEL/','train_on_SRCNNx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        
        logger.info("Prepare Data...")
        data = super().input_setup(imagefolder,self.scale,self.blk_size,self.stride)
        if len(data) == 0:
            logger.info("\nCan Not Find Training Data!\n")
            return

        data_dir = super().get_data_dir("./TRAINED_MODEL/", self.scale)
        data_num = super().get_data_num(data_dir)
        batch_num = data_num // self.batch_size
        
        self.images = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, shape = [None,None,None,self.c_dim], name='labels')
        self.pred = self.srcnn(self.images)
        
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
                                                                           loss_function = super().l2_loss)
        
        self.summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session(config = self.config)
        
        tf.global_variables_initializer().run(session=self.sess)
        
        merged_summary_op = tf.summary.merge_all()
        if os.path.exists('./LOGS/SRCNN'):
            del_file("./LOGS/SRCNN/")
        summary_writer = tf.summary.FileWriter('./LOGS/SRCNN', self.sess.graph)

        test_path = super().prepare_data(validfolder)
        
        flag = -float("Inf")
        logger.info("Now Start Training SRCNN ... ...")
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
                        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["Relu_2"])
                        with tf.gfile.FastGFile(self.save_path+'SRCNN_x{}.pb'.format(self.scale), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
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
                        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["Relu_2"])
                        with tf.gfile.FastGFile(self.save_path+'SRCNN_x{}.pb'.format(self.scale), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        flag = avg_psnr
                    break
        summary_writer.close()
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}).'.format(flag))
        logger.info("--<The training porcess of SRCNN has been completed.>--")
        return "--<The training porcess of SRCNN has been completed.>--"
         
    def test(self,image):
        if image is None:
            raise AttributeError("U must input an image path or an image mtx.")
        elif type(image) == type(np.array([])):
            lr_image = image
        elif type(image) == type(''):
            lr_image = cv.imread(image)
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
        lr_image = lr_image[np.newaxis,:]
        pbPath = "./TRAINED_MODEL/SRCNN_x{}.pb".format(self.scale)
        with tf.Session(config = self.config) as sess:
            with gfile.FastGFile(pbPath,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def,name = '')
            #LR_tensor = sess.graph.get_tensor_by_name("IteratorGetNext:0")
            LR_tensor = sess.graph.get_tensor_by_name("images:0")
            HR_tensor = sess.graph.get_tensor_by_name("Relu_2:0")
            res = sess.run(HR_tensor, feed_dict={LR_tensor: lr_image})
            sess.close()
        res = res[0]
        res = (res + self.mean).clip(min = 0, max = 255)
        res = res.astype(np.uint8)
        return res

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
        
        pbPath = "./TRAINED_MODEL/SRCNN_x{}.pb".format(self.scale)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        
        self.images = self.sess.graph.get_tensor_by_name("IteratorGetNext:0")
        self.pred = self.sess.graph.get_tensor_by_name("Relu_2:0")
        
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

            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/SRCNNx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/SRCNNx{}/".format(self.scale)+data_set))
            #print( "./RESULT/SrOpx{}/".format(self.scale) + fn +"/_{:.4f}_{:.4f}.png" .format(psnr,issim) )
            super().imsave(x, "./RESULT/SRCNNx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )

        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)

