# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:26:40 2021

@author: 月光下的云海
"""
import tensorflow as tf
import numpy as np
import cv2 as cv
from DLL.utils import DegradeFilter,GetPatches,DegradeBic
from glob import glob
from PIL import Image
import os
import h5py

class BaseModel():
    
    """===============================================================================================
    Introduction: Initializtion Function.
    ---------------------------------------------------------------------------------------------------
    Function: __init__
    Nothing.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def __init__(self):
        pass
    
    """===============================================================================================
    Introduction: The leaky relu function.
    ---------------------------------------------------------------------------------------------------
    Function: kernel_init
    Input: x,alpha = 0.6,name = 'lrelu'
    		x		        ----(tf.Variable) the inputed variable.
    		alpha			----(float) the leaky relu factor.
            name            ----(string) Name.
    output:                 ----(tf.Variable) Result.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def lrelu(self,x,alpha = 0.6,name = 'lrelu'):
        return tf.nn.leaky_relu(x,alpha = alpha,name = name)
    
    """===============================================================================================
    Introduction: The function to initialize a kernel.
    ---------------------------------------------------------------------------------------------------
    Function: kernel_init
    Input: shape,name,stddev = 5e-2
    		shape		    ----(int) the shape of kernel.
    		name			----(string) kernel name.
            stddev          ----(float) The variance of initializion numbers.
    output: bais            ----(tf.Variable) the initialized bias.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def kernel_init(self,shape,name,stddev = 5e-2):
        init = tf.truncated_normal_initializer(stddev = 5e-2)
        return(tf.get_variable(shape = shape,initializer = init,name = name))
    
    """===============================================================================================
    Introduction: The function to initialize bias.
    ---------------------------------------------------------------------------------------------------
    Function: bias_init
    Input: shape,name
    		shape		    ----(int) the shape of bias.
    		name			----(string) bias name.
    output: bais            ----(tf.Variable) the initialized bias.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def bias_init(self,shape,name):
        init = tf.constant_initializer(0.0)
        return(tf.get_variable(shape = shape,initializer = init,name = 'bias'))
    
    """===============================================================================================
    Introduction: The function to realize 2D deconvolution.
    ---------------------------------------------------------------------------------------------------
    Function: conv2d
    Input: x,kernel,bias,outshape,strides,padding = 'SAME',act = tf.nn.relu
    		x			    ----(tf.Variable) the input variable.
    		kernel			----(tf.Variable) convolutional kernel.
            bias            ----(tf.Variable) bias.
            outshape        ----(List) [batch_size, image_size1, image_size2, channels]
            strides         ----(int) The strides of convolutional operator.
            padding         ----(String) 'SAME' of 'VALID'. 'SAME' for padding 0 around. 'VALID' for 
                                unpadding around.
            act             ----(Function Handle) The activation function handle.
    output: out             ----(tf.Variable) the output varibale.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def deconv2d(self,x,kernel,bias,outshape,strides,padding = 'SAME',act = tf.nn.relu):
        strides = [1,strides,strides,1]
        y = tf.nn.conv2d_transpose(x,kernel,outshape,strides, padding=padding)
        y = tf.nn.bias_add(y,bias)
        return act(y)
    
    """===============================================================================================
    Introduction: The function to realize 2D convolution.
    ---------------------------------------------------------------------------------------------------
    Function: conv2d
    Input: x, kernel, bias, strides, padding = 'SAME', act = tf.nn.relu
    		x			    ----(tf.Variable) the input variable.
    		kernel			----(tf.Variable) convolutional kernel.
            bias            ----(tf.Variable) bias.
            strides         ----(int) The strides of convolutional operator.
            padding         ----(String) 'SAME' of 'VALID'. 'SAME' for padding 0 around. 'VALID' for 
                                unpadding around.
            act             ----(Function Handle) The activation function handle.
    output: out             ----(tf.Variable) the output varibale.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def conv2d(self,
               x,
               kernel,
               bias,
               strides,
               padding = 'SAME',
               act = tf.nn.relu):
        strides = [1,strides,strides,1]
        conv = tf.nn.conv2d(x,kernel,strides,padding,name = 'conv')
        out = tf.nn.bias_add(conv,bias)
        out = act(out)
        return out
    
    """===============================================================================================
    Introduction: The function to calculate l1 error.
    ---------------------------------------------------------------------------------------------------
    Function: l1_loss
    Input: x,y
    		x			----(tf.Variable) the label of net output.
    		y			----(tf.Variable) outputed variable from net
    output: l1_loss
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def l1_loss(self,x,y):
        return tf.reduce_mean(tf.abs(x-y))
    
    """===============================================================================================
    Introduction: The function to calculate l2 error.
    ---------------------------------------------------------------------------------------------------
    Function: l2_loss
    Input: x,y
    		x			----(tf.Variable) the label of net output.
    		y			----(tf.Variable) outputed variable from net
    output: l2_loss
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def l2_loss(self,x,y):
        return tf.losses.mean_squared_error(x,y)
    
    """===============================================================================================
    Introduction: The function to calculate absolute error.
    ---------------------------------------------------------------------------------------------------
    Function: Abs_loss
    Input: x,y
    		x			----(tf.Variable) the label of net output.
    		y			----(tf.Variable) outputed variable from net
    output: Abs_loss
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def Abs_loss(self,x,y):
        return tf.losses.absolute_difference(x,y)
    
    """===============================================================================================
    Introduction: The function to realize 3D mtx mutipling.
    ---------------------------------------------------------------------------------------------------
    Function: Matmul3D
    Input: y_train_ph,output,batch_size,lr,loss_function = tf.losses.mean_squared_error
    		y_train_ph			----(tf.placehodler) the outputed placeholder of net.
    		output			    ----(tf.Variable) outputed variable from net
    		batch_size		    ----(int) batch size
    		loss_function	    ----(Function Handle) The loss function handle.
    output: train_op,loss
            train_op            ----(list) The trainning operator of loss.
            loss                ----(list) Loss function.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def Matmul3D(self,A,B):
        A_li = [A[:,:,i] for i in range(A.shape[2])]
        B_li = [B[:,:,i] for i in range(B.shape[2])]
        C_li = []
        for a,b in zip(A_li,B_li):
            c = tf.matmul(a,b)
            C_li += [c]
        C = tf.stack(C_li,axis = 2)
        return C
    
    """===============================================================================================
    Introduction: The function to realize learning rate decay.
    ---------------------------------------------------------------------------------------------------
    Function: grenerate_train_op
    Input: y_train_ph,output,batch_size,lr,loss_function = tf.losses.mean_squared_error
    		y_train_ph			----(tf.placehodler) the outputed placeholder of net.
    		output			    ----(tf.Variable) outputed variable from net
    		batch_size		    ----(int) batch size
    		loss_function	    ----(Function Handle) The loss function handle.
    output: train_op,loss
            train_op            ----(list) The trainning operator of loss.
            loss                ----(list) Loss function.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def grenerate_train_op(self,
                           y_train_ph,
                           output,
                           batch_size,
                           lr,
                           lr_decay_rate,
                           var_list = None,
                           loss_function = tf.losses.mean_squared_error):
        loss = loss_function(y_train_ph,output)
        global_step = tf.Variable(0,trainable = False)
        learning_rate = tf.train.exponential_decay(lr,
               global_step,batch_size,lr_decay_rate,staircase = True)
        train_op = tf.train.AdamOptimizer(learning_rate)
        if var_list is None:
            grads,v = zip(*train_op.compute_gradients(loss))
        else:
            grads,v = zip(*train_op.compute_gradients(loss, var_list = var_list))
        grads,_ = tf.clip_by_global_norm(grads,5)
        train_op = train_op.apply_gradients(zip(grads,v),global_step = global_step)
        return train_op,loss,learning_rate 
    
    """===============================================================================================
    Introduction: The function to generate TF-sparse matrix.
    ---------------------------------------------------------------------------------------------------
    Function: GetSparseH
    Input: ker,scale,imshape,is_transpose = False
    		y_train_ph			----(tf.placehodler) the outputed placeholder of net.
    		output			    ----(tf.Variable) outputed variable from net
    		batch_size		    ----(int) batch size
    		loss_function	    ----(Function Handle) The loss function handle.
    output: train_op,loss
            train_op            ----(list) The trainning operator of loss.
            loss                ----(list) Loss function.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def GetSparseH(self,ker,scale,imshape,is_transpose = False):
        s = scale
        lh,lw = imshape
        hh = lh*s
        hw = lw*s
        M = lh*lw
        N = hh*hw
        if ker.shape[0] != ker.shape[1]:
            raise IOError("The kernel is not square with shape {}_{}.".format(ker.shape[0], ker.shape[1]))
        else:
            ws = ker.shape[0]
        t = (ws-1)/2
        cen = int(ws)//2+1
        nv = ws*ws
        nt = nv*M
        R = np.zeros((nt,1),dtype = np.int)
        C = np.zeros((nt,1),dtype = np.int)
        V = np.zeros((nt,1))
        cnt = 1
        pos = np.arange(hh*hw)
        pos = pos.reshape((hh,hw),order = 'F')
        for lrow in range(1,lh+1):
            for lcol in range(1,lw+1):
                row = (lrow-1)*s+1
                col = (lcol-1)*s+1
                row_idx = (lcol-1)*lh+lrow
                rmin = int(max(row-t,1))
                rmax = int(min(row+t,hh))
                cmin = int(max(col-t,1))
                cmax = int(min(col+t,hw))
                sup = pos[rmin-1:rmax,cmin-1:cmax]
                col_ind = sup.reshape((-1,1),order = 'F')
                r1 = row-rmin
                r2 = rmax-row
                c1 = col-cmin
                c2 = cmax-col
                ker2 = ker[cen-r1-1:cen+r2,cen-c1-1:cen+c2]
                ker2 = ker2.reshape((-1,1),order = 'F')
                nn = col_ind.shape[0]
                R[cnt-1:cnt+nn-1] = row_idx-1
                C[cnt-1:cnt+nn-1] = col_ind
                V[cnt-1:cnt+nn-1] = ker2/sum(ker2)
                cnt += nn
        R = R[:cnt-1].reshape((-1),order = 'F')
        C = C[:cnt-1].reshape((-1),order = 'F')
        V = V[:cnt-1].reshape((-1),order = 'F')
        if not is_transpose:
            indices = [[r,c] for r,c in zip(R,C)]
            return tf.SparseTensor(indices=indices, values=list(V), dense_shape=[M, N])
        else:
            indices = [[c,r] for c,r in zip(C,R)]
            return tf.SparseTensor(indices=indices, values=list(V), dense_shape=[N, M])
    
    """===============================================================================================
    Introduction: The function to convert sparse matrix into sparse tensor.
    ---------------------------------------------------------------------------------------------------
    Function: SparseMTX2SparseTensor
    Input: A
    		A			        ----(scipy.coo) The coo sparse matrix.
    output:                     ----(tf.SparseTensor) tensorflow sparse tensor
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def SparseMTX2SparseTensor(self,A):
        from scipy.sparse import find
        row_id,col_id,val = find(A)
        indices = [[r,c] for r,c in zip(row_id,col_id)]
        return tf.SparseTensor(indices=indices, values=list(val), dense_shape=A.shape)
    
    """===============================================================================================
    Introduction: The function to calculate model parameters.
    ---------------------------------------------------------------------------------------------------
    Function: SparseMTX2SparseTensor
    Input: A
    		A			        ----(scipy.coo) The coo sparse matrix.
    output:                     ----(tf.SparseTensor) tensorflow sparse tensor
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def count_param(self,variables):
        #variables = tf.trainable_variables()
        total_parameters = 0
        for variable in variables:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        return total_parameters
    
    """===============================================================================================
    Introduction: The function to generate the list of train data.
    ---------------------------------------------------------------------------------------------------
    Function: GetDataGroup
    Input: x_data,y_data,blk_size,scale,batch_size = 50,ch,is_restore,is_norm = True,
           train_rate = None,is_shuffle = True
    		x_data			----(numpy) inputed data
    		y_data			----(numpy) outputed data
            blk_size        ----(int) block size
            scale           ----(int) Scaling factor
    		batch_size		----(int) batch size
    		is_norm	        ----(BOOL) If it is True, the function would normalize data.
                                       Else, the data wouldn't be normalized.
    		train_rate	    ----(int) the rate of train data size
            is_shuffle      ----(BOOL) If it is True, the input data would be shuffled.
    output: xtrain_li,ytrain_li,xtest_li,ytest_li
            xtrain_li       ----(list) The list of inputed train batch.
            ytrain_li       ----(list) The list of outputed train batch.
            xtest_li        ----(list) The list of inputed test batch.
            ytest_li        ----(list) The list of ouputed test batch.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    "==============================================================================================="""
    def get_data_group(self,
                       x_data,
                       y_data,
                       blk_size,
                       scale,
                       batch_size = 50,
                       ch = 3,
                       is_restore = True,
                       is_norm = True,
                       train_rate = None,
                       is_shuffle = True):
        if is_norm:
            x_data = x_data/255
            y_data = y_data/255
        if train_rate is None:
            #raise ZeroDivisionError("U must intput train_rate.")
            train_rate = 1
        else:
            pass
        total_size = x_data.shape[0]
        #Shuffle
        total_index = np.arange(total_size)
        if is_shuffle: np.random.shuffle(total_index)
        #Split into train and  test
        train_index = total_index[:int(total_size*train_rate)]
        test_index = total_index[int(total_size*train_rate):]
        xtrain,ytrain = x_data[train_index],y_data[train_index]
        xtest,ytest = x_data[test_index],y_data[test_index]
        #Grouping
        train_size,test_size = xtrain.shape[0],xtest.shape[0]
        train_end = train_size if train_size%batch_size == 0 else (train_size//batch_size)*batch_size
        test_end = test_size if test_size%batch_size == 0 else (test_size//batch_size)*batch_size
        xtrain,ytrain,xtest,ytest = xtrain[:train_end],ytrain[:train_end],xtest[:test_end],ytest[:test_end]
        xtrain_li = np.vsplit(xtrain.reshape((-1,blk_size,blk_size,ch)),train_end//batch_size)
        if is_restore:
            ytrain_li = np.vsplit(ytrain.reshape((-1,blk_size,blk_size,ch)),train_end//batch_size)
        else:
            ytrain_li = np.vsplit(ytrain.reshape((-1,blk_size*scale,blk_size*scale,ch)),train_end//batch_size)
        try:
            xtest_li = np.vsplit(xtest.reshape((-1,blk_size,blk_size,ch)),test_end//batch_size)
            if is_restore:
                ytest_li = np.vsplit(ytest.reshape((-1,blk_size*scale,blk_size*scale,ch)),test_end//batch_size)
            else:
                ytest_li = np.vsplit(ytest.reshape((-1,blk_size*scale,blk_size*scale,ch)),test_end//batch_size)
        except ZeroDivisionError:
            xtest_li = None
            ytest_li = None
        return xtrain_li,ytrain_li,xtest_li,ytest_li
    
    # def make_dataset2(self,paths,scale,blk_size,step):
    #     fn_li = glob(paths+'*.png')
    #     Phr_li = []
    #     Plr_li = []
    #     for fn in fn_li:
    #         img = cv.imread(fn)
    #         Phr,_ = GetPatches(img,blk_size,step,channel_order = 'lbb',need_flatten = False,completeness = False)
    #         Phr = np.uint8(Phr)
    #         Phr_li += [Phr]
    #         lr_img = DegradeBic(img,scale,is_restore = False)
    #         Plr,_ = GetPatches(lr_img,blk_size//scale,step//scale,channel_order = 'lbb',need_flatten = False,completeness = False)
    #         Plr = np.uint8(Plr)
    #         Plr_li += [Plr]
    #     nPlr = np.vstack(Plr_li)
    #     del Plr_li,Plr
    #     nPhr = np.vstack(Phr_li)
    #     del Phr_li,Phr
    #     return nPlr,nPhr
    
    """===============================================================================================
    Introduction: Python generator-style dataset. Creates low-res and corresponding high-res patches.
    ---------------------------------------------------------------------------------------------------
    Function: make_dataset
    Input: paths, scale, mean, size_lr
    		paths			----(List) The list of inputed image path.
    		scale			----(numpy) Scaling factor
            mean            ----(int) mean of inputed data
    		size_lr		    ----(int) size of lr image.
    output: x,y
            x               ----(numpy) Numpy matrix of inputed image.
            y               ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def make_dataset(self,paths, scale, mean, size_lr):
        
        size_hr = size_lr * scale
        for p in paths:
            
            fn = os.path.split(p)[-1]
            fn = os.path.splitext(fn)[0]
            fn = os.path.join('./DATABSE/DIV2K_BlurKernel/',fn,fn+'_kernel_x2.mat')
            import scipy.io as scio
            ker = scio.loadmat(fn)
            ker = np.array(ker['Kernel'])
            
            im = cv.imread(p.decode(), 3)
            hr = im[0:(im.shape[0] - (im.shape[0] % scale)),
                      0:(im.shape[1] - (im.shape[1] % scale)), :]
            lr = Image.fromarray(hr.astype(dtype = np.uint8))
            
            lr = DegradeFilter(lr,blur_kernel = ker)
            # lr = lr.resize((lr.size[0]//scale,lr.size[1]//scale) ,Image.BICUBIC )
            lr = (np.array(lr)).astype(dtype=np.uint8)
            
            # im = Image.open(p)
            # hr = im.crop( (0,0,im.size[0]-im.size[0]%scale,im.size[1]-im.size[1]%scale) )
            # lr = hr.resize((hr.size[0]//scale,hr.size[1]//scale) ,Image.BICUBIC )
            
            # hr = np.array(hr)-mean
            # lr = np.array(lr)-mean
            
            # hr = np.array(hr).astype(np.uint8)-mean
            # lr = np.array(lr).astype(np.uint8)-mean
            
            numx = int(lr.shape[0] / size_lr)
            numy = int(lr.shape[1] / size_lr)
            
            # # normalize
            # im_norm = cv.imread(p.decode(), 3).astype(np.float32) - mean
            
            # #im_norm = cv.imread(p, 3).astype(np.float32) - mean
            # # random flip
            # import random
            # r = random.randint(-1, 2)
            # if not r == 2:
            #     im_norm = cv.flip(im_norm, r)
            # # divisible by scale - create low-res
            # hr = im_norm[0:(im_norm.shape[0] - (im_norm.shape[0] % scale)),
            #           0:(im_norm.shape[1] - (im_norm.shape[1] % scale)), :]
            # lr = cv.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
            #                 interpolation=cv.INTER_CUBIC)
            
            # print(lr.shape)
    
            # numx = int(lr.shape[0] / size_lr)
            # numy = int(lr.shape[1] / size_lr)
    
            for i in range(0, numx):
                startx = i * size_lr
                endx = (i * size_lr) + size_lr
    
                startx_hr = i * size_hr
                endx_hr = (i * size_hr) + size_hr
    
                for j in range(0, numy):
                    starty = j * size_lr
                    endy = (j * size_lr) + size_lr
                    starty_hr = j * size_hr
                    endy_hr = (j * size_hr) + size_hr
    
                    crop_lr = lr[startx:endx, starty:endy]
                    crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]
    
                    x = crop_lr.reshape((size_lr, size_lr, 3))
                    y = crop_hr.reshape((size_hr, size_hr, 3))
                    yield x/255.0, y/255.0
    
    """===============================================================================================
    Introduction: Python generator-style dataset. Creates blur low-res and corresponding high-res patches.
    ---------------------------------------------------------------------------------------------------
    Function: make_blur_dataset
    Input: paths,scale,ker,size_lr
    		paths			----(List) The list of inputed image path.
    		scale			----(numpy) Scaling factor
            ker             ----(int) blur kernel.
    		size_lr		    ----(int) size of lr image.
    output: x,y
            x               ----(numpy) Numpy matrix of inputed image.
            y               ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def make_blur_dataset(self,paths,scale,ker,size_lr):
        size_hr = size_lr * scale
        for p in paths:
            # normalize
            im_norm = cv.imread(p.decode(), 3).astype(np.float32)
            #im_norm = cv.imread(p, 3).astype(np.float32) - mean
            # random flip
            import random
            r = random.randint(-1, 2)
            if not r == 2:
                im_norm = cv.flip(im_norm, r)
            # divisible by scale - create low-res
            hr = im_norm[0:(im_norm.shape[0] - (im_norm.shape[0] % scale)),
                      0:(im_norm.shape[1] - (im_norm.shape[1] % scale)), :]
            lr = DegradeFilter(hr,blur_kernel = ker)
            lr = cv.resize(lr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
                            interpolation=cv.INTER_CUBIC)
            
            numx = int(lr.shape[0] / size_lr)
            numy = int(lr.shape[1] / size_lr)
    
            for i in range(0, numx):
                startx = i * size_lr
                endx = (i * size_lr) + size_lr
    
                startx_hr = i * size_hr
                endx_hr = (i * size_hr) + size_hr
    
                for j in range(0, numy):
                    starty = j * size_lr
                    endy = (j * size_lr) + size_lr
                    starty_hr = j * size_hr
                    endy_hr = (j * size_hr) + size_hr
    
                    crop_lr = lr[startx:endx, starty:endy]
                    crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]
    
                    x = crop_lr.reshape((size_lr, size_lr, 3))
                    y = crop_hr.reshape((size_hr, size_hr, 3))
                    yield x, y
    
    '''
    def make_val_dataset3(self,paths, scale, mean):
        """
        Python generator-style dataset for the validation set. Creates input and ground truth.
        """
        for p in paths:
            
            im = cv.imread(p.decode(), 3)
            hr = im[0:(im.shape[0] - (im.shape[0] % scale)),
                      0:(im.shape[1] - (im.shape[1] % scale)), :]
            lr = Image.fromarray(hr.astype(dtype = np.uint8))
            lr = lr.resize((lr.size[0]//scale,lr.size[1]//scale) ,Image.BICUBIC )
            lr = (np.array(lr)).astype(dtype=np.uint8)
            
            # im = Image.open(p)
            # hr = im.crop( (0,0,im.size[0]-im.size[0]%scale,im.size[1]-im.size[1]%scale) )
            # lr = hr.resize((hr.size[0]//scale,hr.size[1]//scale), Image.BICUBIC )
            
            # hr = np.array(hr)-mean
            # lr = np.array(lr)-mean
            
            # normalize
            # im_norm = cv.imread(p.decode(), 3).astype(np.float32) - mean
            #im_norm = cv.imread(p,3).astype(np.float32) - mean
            # divisible by scale - create low-res
            # hr = im_norm[0:(im_norm.shape[0] - (im_norm.shape[0] % scale)),
            #           0:(im_norm.shape[1] - (im_norm.shape[1] % scale)), :]
            # lr = cv.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
            #                 interpolation=cv.INTER_CUBIC)
            yield lr/255.0, hr/255.0
    
    
    def make_val_dataset2(self,paths,scale):
        paths = glob(paths+'*.bmp')
        for p in paths:
            img = cv.imread(p)
            lr =  DegradeBic(img,scale,is_restore = False)
            lr = lr.reshape((1,)+lr.shape)
            img = img.reshape((1,)+img.shape)
            yield lr,img
    '''
    
    """===============================================================================================
    Introduction: The function to make blur valuation dataset.
    ---------------------------------------------------------------------------------------------------
    Function: make_blur_val_dataset
    Input: paths,scale,ker
    		paths			----(List) The list of inputed image path.
    		scale			----(numpy) Scaling factor
            ker             ----(int) blur kernel.
    output: lr, hr
            lr              ----(numpy) Numpy matrix of inputed image.
            hr              ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def make_blur_val_dataset(self,paths,scale,ker):
        for p in paths:
            # normalize
            im_norm = cv.imread(p.decode(), 3).astype(np.float32)
            #im_norm = cv.imread(p,3).astype(np.float32) - mean
            # divisible by scale - create low-res
            hr = im_norm[0:(im_norm.shape[0] - (im_norm.shape[0] % scale)),
                      0:(im_norm.shape[1] - (im_norm.shape[1] % scale)), :]
            lr = DegradeFilter(hr,blur_kernel = ker)
            lr = cv.resize(lr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
                            interpolation=cv.INTER_CUBIC)
            yield lr, hr
    
    """===============================================================================================
    Introduction: The function to prepare data.
    ---------------------------------------------------------------------------------------------------
    Function: prepare_data
    Input: train_set
    		train_set		----(string) The inputed image path.
    output: data
            data            ----(list) The list of inputed image paths.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def prepare_data(self,train_set):
        data = glob(train_set)
        return data
    
    """===============================================================================================
    Introduction: The function to setup input.
    ---------------------------------------------------------------------------------------------------
    Function: input_setup
    Input: train_set,scale,image_size,stride
    		train_set			----(string) The inputed image path.
    		scale			    ----(int) Scaling factor
            image_size          ----(int) The inputed lr image size.
            stride              ----(int) The croped stride.
    output: data
            data                ----(List) The list of inputed image paths.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def input_setup(self,train_set,scale,image_size,stride):
        """
            Read image files and make their sub-images and saved them as a h5 file format
        """
    
        data = self.prepare_data(train_set)
        self.make_sub_data(data, scale, image_size, stride)
        return data
    
    """===============================================================================================
    Introduction: The function to make valuation dataset.
    ---------------------------------------------------------------------------------------------------
    Function: make_val_dataset
    Input: path, scale
    		path			----(List) The list of inputed image path.
    		scale			----(int) Scaling factor
    output: lr, hr
            lr              ----(numpy) Numpy matrix of inputed image.
            hr              ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def make_val_dataset(self,path, scale):
        eng = None
        mdouble = None
        for p in path:
            image, label = self.preprocess(p, scale, eng, mdouble)
            # image = image[np.newaxis, :]
            # label = label[np.newaxis, :]
            yield image/255.0, label/255.0
    
    """===============================================================================================
    Introduction: The function to make sub-image.
    ---------------------------------------------------------------------------------------------------
    Function: make_sub_data
    Input: data, scale, image_size, stride, c_dim = 3
    		data			----(List) The list of inputed image path.
    		scale			----(int) Scaling factor
            image_size      ----(int) blur kernel.
            stride          ----(int) The stride of croped image.
            c_dim           ----(int) The inputed channels.
    output: sub_input,sub_label
            sub_input       ----(numpy) Numpy matrix of inputed image.
            sub_label       ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def make_sub_data(self,data, scale, image_size, stride, c_dim = 3):

        for i in range(len(data)):
            input_, label_, = self.preprocess(data[i], scale, None, None)
            if len(input_.shape) == 3:
                h, w, c = input_.shape
            else:
                h, w = input_.shape
    
            for x in range(0, h * scale - image_size * scale + 1, stride * scale):
                for y in range(0, w * scale - image_size * scale + 1, stride * scale):
                    sub_label = label_[x: x + image_size * scale, y: y + image_size * scale]
                    
                    sub_label = sub_label.reshape([image_size * scale , image_size * scale, c_dim])
    
                    t = cv.cvtColor(sub_label, cv.COLOR_BGR2YCR_CB)
                    t = t[:, :, 0]
                    gx = t[1:, 0:-1] - t[0:-1, 0:-1]
                    gy = t[0:-1, 1:] - t[0:-1, 0:-1]
                    Gxy = (gx**2 + gy**2)**0.5
                    r_gxy = float((Gxy > 10).sum()) / ((image_size*scale)**2) * 100
                    if r_gxy < 10:
                        continue
    
                    sub_label =  sub_label / 255.0
    
                    x_i = int(x / scale)
                    y_i = int(y / scale)
                    sub_input = input_[x_i: x_i + image_size, y_i: y_i + image_size]
                    sub_input = sub_input.reshape([image_size, image_size, c_dim])
                    sub_input = sub_input / 255.0
    
                    
                    yield sub_input,sub_label
    
    """===============================================================================================
    Introduction: The function to preprocess data.
    ---------------------------------------------------------------------------------------------------
    Function: make_blur_val_dataset
    Input: path, scale = 3, eng = None, mdouble = None
    		path			----(string) The inputed image path.
    		scale			----(int) Scaling factor
            mdouble         ----(string) the bicubic model.
    output: input_, label_
            input_          ----(numpy) Numpy matrix of inputed image.
            label_          ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def preprocess(self,path, scale = 3, eng = None, mdouble = None):
        img = self.imread(path)
        label_ = self.modcrop(img, scale)
        if eng is None:
            # input_ = cv2.resize(label_, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC)
            input_ = self.PIL_resize(label_, 1.0/scale, Image.BICUBIC)
        else:
            input_ = np.asarray(eng.imresize(mdouble(label_.tolist()), 1.0/scale, 'bicubic'))
    
        input_ = input_[:, :, ::-1]
        label_ = label_[:, :, ::-1]
    
        return input_, label_
    
    """===============================================================================================
    Introduction: The function to make resize image by PIL.
    ---------------------------------------------------------------------------------------------------
    Function: PIL_resize
    Input: image, ratio, mode
    		image			----(numpy) The image matrix.
    		ratio			----(numpy) Scaling factor
            mode            ----(int) Image.Bicubic.
    output: image_resize
            image_resize    ----(numpy) The resized image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def PIL_resize(self,image, ratio, mode):
        PIL_image = Image.fromarray(image.astype(dtype=np.uint8))
        PIL_image_resize = PIL_image.resize((int(PIL_image.size[0] * ratio), int(PIL_image.size[1] * ratio)), mode)
        image_resize = (np.array(PIL_image_resize)).astype(dtype=np.uint8)
        return image_resize
    
    """===============================================================================================
    Introduction: The function to crop image.
    ---------------------------------------------------------------------------------------------------
    Function: modcrop
    Input: img, scale =3
    		img			    ----(numpy) The image matrix.
    		scale			----(int) Scaling factor
    output: img
            img             ----(numpy) The croped image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def modcrop(self,img, scale =3):
        if len(img.shape) ==3:
            h, w, _ = img.shape
            h = int((h / scale)) * scale
            w = int((w / scale)) * scale
            img = img[0:h, 0:w, :]
        else:
            h, w = img.shape
            h = int((h / scale)) * scale
            w = int((w / scale)) * scale
            img = img[0:h, 0:w]
        return img
    
    """===============================================================================================
    Introduction: The function to read image.
    ---------------------------------------------------------------------------------------------------
    Function: imread
    Input: path
    		path			----(List) The inputed image path.
    output: img
            img             ----(numpy) Numpy matrix of readed image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def imread(self,path):
        img =  cv.imread(path)
        return img
    
    """===============================================================================================
    Introduction: The function to save image.
    ---------------------------------------------------------------------------------------------------
    Function: imsave
    Input: image, path
    		image			----(numpy) The image matrix.
    		path			----(string) The inputed image path.
    output: None
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def imsave(self, image, path):
        cv.imwrite(os.path.join(os.getcwd(),path),image)
    
    """===============================================================================================
    Introduction: The function to make h5 file.
    ---------------------------------------------------------------------------------------------------
    Function: make_data_hf
    Input: input_, label_, times, image_size, scale, checkpoint_dir, c_dim = 3
    		input_			  ----(numpy) The inputed image matrix.
    		label_			  ----(numpy) The label image matrix.
            times             ----(int) Times.
            image_size        ----(int) The lr image size.
            scale             ----(int) The scaling factor.
            checkpoint_dir    ----(string) saving path.
            c_dim             ----(int) Image channels.
    output: True
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def make_data_hf(self,input_, label_, times, image_size, scale, checkpoint_dir, c_dim = 3):
        if not os.path.isdir(os.path.join(os.getcwd(),checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(),checkpoint_dir))
    
        savepath = os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'train_x%d.h5' % scale)
        
        if times == 0:
            if os.path.exists(savepath):
                print("\n%s have existed!\n" % (savepath))
                return False
            else:
                hf = h5py.File(savepath, 'w')
    
                input_h5 = hf.create_dataset("input", (1, image_size, image_size, c_dim), 
                                            maxshape=(None, image_size, image_size, c_dim), 
                                            chunks=(1, image_size, image_size, c_dim), dtype='float32')
                label_h5 = hf.create_dataset("label", (1, image_size*scale, image_size*scale, c_dim), 
                                            maxshape=(None, image_size*scale, image_size*scale, c_dim), 
                                            chunks=(1, image_size*scale, image_size*scale, c_dim),dtype='float32')
        else:
            hf = h5py.File(savepath, 'a')
            input_h5 = hf["input"]
            label_h5 = hf["label"]
    
        input_h5.resize([times + 1, image_size, image_size, c_dim])
        input_h5[times : times+1] = input_
        label_h5.resize([times + 1, image_size*scale, image_size*scale, c_dim])
        label_h5[times : times+1] = label_
        hf.close()
        return True
    
    """===============================================================================================
    Introduction: The function to rotate image randomly.
    ---------------------------------------------------------------------------------------------------
    Function: augmentation
    Input: batch, random
    		batch			----(int) batch size.
    		random			----(float) random number.
    output: batch_rot       ----(numpy) Rotated image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def augmentation(self,batch, random):
        if random[0] < 0.3:
            batch_flip = np.flip(batch, 1)
        elif random[0] > 0.7:
            batch_flip = np.flip(batch, 2)
        else:
            batch_flip = batch
    
        if random[1] < 0.5:
            batch_rot = np.rot90(batch_flip, 1, [1, 2])
        else:
            batch_rot = batch_flip
    
        return batch_rot
    
    """===============================================================================================
    Introduction: The function to get h5 dir.
    ---------------------------------------------------------------------------------------------------
    Function: get_data_dir
    Input: checkpoint_dir, scale
    		checkpoint_dir	----(string) The h5 dir.
    		scale			----(int) Scaling factor
    output:                 ----(string) h5 dir.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def get_data_dir(self,checkpoint_dir, scale):
        return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'train_x%d.h5' % scale)
    
    """===============================================================================================
    Introduction: The function to get data length.
    ---------------------------------------------------------------------------------------------------
    Function: get_data_num
    Input: path
    		path			----(string) The h5 path.
    output:                 ----(int) data length.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def get_data_num(self,path):
        with h5py.File(path, 'r') as hf:
           input_ = hf['input']
           return input_.shape[0]
    
    """===============================================================================================
    Introduction: The function to get batch data.
    ---------------------------------------------------------------------------------------------------
    Function: get_batch
    Input: path, data_num, batch_size
    		path			----(string) The inputed image path.
    		data_num		----(int) data length
            batch_size      ----(int) batch_size
    output: batch_images, batch_labels
            batch_images    ----(numpy) Numpy matrix of inputed image.
            batch_labels    ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def get_batch(self,path, data_num, batch_size):
        with h5py.File(path, 'r') as hf:
            input_ = hf['input']
            label_ = hf['label']
    
            random_batch = np.random.rand(batch_size) * (data_num - 1)
            batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
            batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
            for i in range(batch_size):
                batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
                batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])
    
            random_aug = np.random.rand(2)
            batch_images = self.augmentation(batch_images, random_aug)
            batch_labels = self.augmentation(batch_labels, random_aug)
            return batch_images, batch_labels
    
    """===============================================================================================
    Introduction: The function to get processed image.
    ---------------------------------------------------------------------------------------------------
    Function: get_image
    Input: path, scale, matlab_bicubic
    		path			    ----(string) The inputed image path.
    		scale			    ----(int) Scaling factor
            matlab_bicubic      ----(BOOL) True or False. True for matlab bicubic. False for PIL resize
    output: image, label
            image               ----(numpy) Numpy matrix of inputed image.
            label               ----(numpy) Numpy matrix of label image.
    ---------------------------------------------------------------------------------------------------
    Created on Sat Nov 21 10:54:59 2020
    @author: 月光下的云海
    ================================================================================================"""
    def get_image(self, path, scale, matlab_bicubic):
        if matlab_bicubic:
            import matlab.engine
            eng = matlab.engine.start_matlab()
            mdouble = matlab.double
        else:
            eng = None
            mdouble = None
        
        image, label = self.preprocess(path, scale, eng, mdouble)
        image = image[np.newaxis, :]
        label = label[np.newaxis, :]
    
        if matlab_bicubic:
            eng.quit()
    
        return image, label
    
    