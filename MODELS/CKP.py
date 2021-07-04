# -*- coding: utf-8 -*-
"""===============================================================================================
The Python code of SrOp
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
Created on Sat May  1 14:07:30 2021
@author: 月光下的云海(西电博巍)
Version: Ultimate
==============================================================================================="""

import tensorflow.contrib.slim as slim
import tensorflow as tf
from .BaseModel import BaseModel
import numpy as np
import cv2 as cv
from time import time
import os
from DLL.utils import DegradeFilter,GetPatches,del_file
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import scipy.io as scio

class CKP(BaseModel):
    
    def __init__(self,scale = 4,epoch = 10000):
        self.scale = 1
        self.lr = 1e-3
        self.batch_size = 16
        self.epochs = epoch
        self.blk_size = 32
        self.save_path = './TRAINED_MODEL/'
        self.remark = "U need to enlarge the picture to the specified size first."
        self.ch = 3
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        
    def GlobalPooling(self,y):
        return tf.reduce_mean(y, [1, 2], keep_dims=True, name='GAP')
    
    def Predictor(self,x,scope = 'Predictor',reuse = None):
        #l = x.get_shape().as_list()[0]
        l = tf.shape(x)[0:1]
        print(l)
        with tf.variable_scope(scope,reuse = reuse):
            with slim.arg_scope([slim.conv2d],activation_fn = self.lrelu):
                y = slim.conv2d(x,64,5,stride = 1,padding = 'VALID',scope = 'conv1')
                y = slim.conv2d(y,128,5,stride = 1,padding = 'VALID',scope = 'conv2')
                y = slim.conv2d(y,256,5,stride = 1,padding = 'VALID',scope = 'conv3')
                y = slim.conv2d(y,512,5,stride = 1,padding = 'VALID',scope = 'conv4')
                y = slim.conv2d(y,512,5,stride = 1,padding = 'VALID',scope = 'conv5')
                y = slim.conv2d(y,17*17,5,stride = 1,padding = 'VALID',scope = 'conv6',activation_fn = tf.identity)
            y = self.GlobalPooling(y)
            y = tf.squeeze(y)
            #y = tf.reshape(y,[l,17,17])
            nsh = tf.concat([l,tf.constant([17,17])],axis = 0)
            y = tf.reshape(y,nsh)
            return y
    
    def Degard(self,x,ker):
        ker_li = tf.split(ker,self.batch_size,axis = 0)
        x_li = tf.split(x,self.batch_size,axis = 0)
        degard_im_li = []
        for x,ker in zip(x_li,ker_li):
            ker = tf.squeeze(ker)
            sh1 = tf.concat([tf.shape(ker),tf.constant([1])],axis = 0)
            nker = tf.concat([tf.reshape(ker,sh1) for _ in range(self.ch)],axis = 2)
            # nker = tf.concat([tf.reshape(ker,list(ker.shape)+[1,]) for _ in range(self.ch)],axis = 2)
            sh2 = tf.concat([tf.shape(nker),tf.constant([1])],axis = 0)
            nker = tf.concat([tf.reshape(nker,sh2) for _ in range(self.ch)],axis = 3)
            #nker = tf.concat([tf.reshape(nker,list(nker.shape)+[1,]) for _ in range(self.ch)],axis = 3)
            degard_im = tf.nn.conv2d(x, nker, strides = 1, padding = 'SAME')
            degard_im_li += [degard_im]
        return tf.concat(degard_im_li,axis = 0)
    
    def make_dataset(self,scale,blk_size,step,imagefolder,ker_root):
        dir_li = os.listdir(ker_root)
        Phr_li = []
        Plr_li = []
        ker_li = []
        for dirn in dir_li:
            #print(dirn)
            for _,_,files in os.walk(os.path.join(ker_root,dirn)):
                fn = os.path.join(ker_root,dirn,files[0])
                ker = scio.loadmat(fn)
                ker = np.array(ker['Kernel'])
            img = cv.imread(imagefolder+dirn+'.png')
            Phr,_ = GetPatches(img,blk_size,step,need_flatten = False,completeness = False)
            Phr = np.uint8(Phr)
            Phr_li += [Phr]
            del Phr
            lr_img = DegradeFilter(img,blur_kernel = ker)
            ker = ker.reshape((1,)+ker.shape)
            Plr,_ = GetPatches(lr_img,blk_size//scale,step//scale,need_flatten = False,completeness = False)
            Plr = np.uint8(Plr)
            Plr_li += [Plr]
            ker = np.concatenate([ker for _ in range(Plr.shape[0])],axis = 0)
            del Plr
            ker_li += [ker]
            del ker
        nPlr = np.vstack(Plr_li)
        del Plr_li
        nPhr = np.vstack(Phr_li)
        del Phr_li
        nker = np.vstack(ker_li)
        del ker_li
        return nPlr,nPhr,nker
    
    def train(self,imagefolder, ker_root):
        
        # Preprocessing Image Data
        print("Preprocessing Image Data... ...")
        x_data,y_data,ker_data = self.make_dataset(self.scale,self.blk_size,self.blk_size*self.scale,imagefolder, ker_root)
        
        # Placeholder and Loss
        input_ph = tf.placeholder(shape = [None,None,None,self.ch],dtype = tf.float32,name = 'InputPh')
        label_im = tf.placeholder(shape = [None,None,None,self.ch],dtype = tf.float32,name = 'LabelPh')
        label_ker = tf.placeholder(shape = [None,17,17],dtype = tf.float32)
        ker_map = self.Predictor(input_ph)
        degard_im = self.Degard(x = label_im,ker = ker_map)
        print("\nThe parameters volumn is:")
        print(super().count_param(tf.global_variables()))
        loss_im = super().l1_loss(degard_im,input_ph)
        loss_ker = super().l2_loss(ker_map,label_ker)
        loss = loss_ker+0.0001*loss_im
        train_op = self.GenerateTrainOp(loss)
        
        # Train CKP
        print('Training CKP ... ...')
        sess = tf.Session(config = self.config)
        sess.run(tf.global_variables_initializer())
        flag = float('Inf')
        l_li = []
        l_ker_li = []
        if os.path.exists('./LOGS/CKP/'):
            del_file("./LOGS/CKP/")
        train_writer = tf.summary.FileWriter('./LOGS/CKP', sess.graph)
        for e in range(self.epochs):
            batch1 = np.random.randint(0,len(x_data),size = self.batch_size)
            x_train = x_data[batch1]
            y_train = y_data[batch1]
            ker_train = ker_data[batch1]
            batch2 = np.random.randint(0,len(x_data),size = self.batch_size)
            x_test = x_data[batch2]
            y_test = y_data[batch2]
            ker_test = ker_data[batch2]
            feed_dict = {input_ph:x_train,label_im:y_train,label_ker:ker_train}
            _,train_loss,loss1,loss2 = sess.run([train_op,loss,loss_ker,loss_im],feed_dict = feed_dict)
            if e % 100 == 99:
                start_t = time()
                feed_dict = {input_ph:x_test,label_im:y_test,label_ker:ker_test}
                test_loss = sess.run(loss,feed_dict = feed_dict)
                print('[{:.3f}]Epoch:{},train_loss:{:.5f},test_loss:{:.5f}'.format(time()-start_t,e+1,train_loss,test_loss))
                l_li += [loss1+loss2]
                l_ker_li += [loss1]
                if train_loss<flag:
                    print('Saving the beter model')
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Predictor/Reshape"])
                    with tf.gfile.FastGFile(self.save_path+'CKP.pb', mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
                    flag = train_loss
        train_writer.close()
        sess.close()

    def GenerateTrainOp(self,loss):
        global_step = tf.Variable(0,trainable = False)
        learning_rate = tf.train.exponential_decay(self.lr,global_step,self.batch_size,0.9,staircase = True)
        train_op = tf.train.AdamOptimizer(learning_rate)
        grads,v = zip(*train_op.compute_gradients(loss))
        grads,_ = tf.clip_by_global_norm(grads,5)
        train_op = train_op.apply_gradients(zip(grads,v),global_step = global_step)
        return train_op
    
    def test(self,image):
        if image is None:
            raise AttributeError("U must input an image path or an image mtx.")
        elif type(image) == type(np.array([])):
            lr_image = image
        elif type(image) == type(''):
            lr_image = cv.imread(image)
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
        lr_image = lr_image.reshape((1,)+lr_image.shape)
        pbPath = "./TRAINED_MODEL/CKP.pb"
        sess = tf.Session()
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = sess.graph.get_tensor_by_name("InputPh:0")
        ker_map = sess.graph.get_tensor_by_name("Predictor/Reshape:0")
        t = time()
        res = sess.run(ker_map, feed_dict={LR_tensor: lr_image})
        sess.close()
        res = res[0]
        print("Time Elapsed:",time()-t)
        return res
    