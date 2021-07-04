# -*- coding: utf-8 -*-


import numpy as np
import cv2 as cv
#import tensorflow as tf
import logging
import os
import matplotlib
matplotlib.use('Agg')
import collections
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

"""===============================================================================================
Introduction: Basic math function.
---------------------------------------------------------------------------------------------------
Function: 
    GetAbsMax     ---- get max abs value.
    GetAbsMin     ---- get min abs value.
    ave           ---- get average value.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
GetAbsMax = lambda x:np.abs(x).max()
GetAbsMin = lambda x:np.abs(x).min()
ave = lambda x:sum(x)/len(x)

"""===============================================================================================
Introduction: The functin to delete file.
---------------------------------------------------------------------------------------------------
Function: del_file
Input: path
		path			----(string) the path to delete.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
    os.removedirs(path)

"""===============================================================================================
Introduction: The functin to block an image into patches. The idea is quoted from W.S. Dong NCSR
(Dong W , Zhang L , Shi G . Nonlocally Centralized Sparse Representation for Image Restoration[J]. 
IEEE Transactions on Image Processing, 2013, 22(4).)
---------------------------------------------------------------------------------------------------
Function: GetPatches
Input: image,b,s,need_flatten = True,completeness = False
		image			----(numpy) inputed image
		b				----(int) block size
		s				----(int) block step
        channel_order   ----(string) 'bbl' or 'lbb'. 
                            If 'bbl', the patches would with size (blk_size,blk_size,length).
                            If 'lbb', the pathces would with size (length, blk_size, blk_size).
		need_flatten	----(BOOL) If it is True, the function would flatten every image patche into 
							the flattened vector. Else, the patches would be stacked straightly.
		completeness	----(BOOL) If it is True, the function would block image with stride=1 at the 
							end of image, so as to avoid miss info which out of step. Else, the function 
							would straightly droup out the out-step info.
Return: Px,ch
        Px              ----(numpy) The block image mtx.
        ch              ----(int) Channels.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def GetPatches(image,
               b,
               s,
               channel_order = 'bbl',#'bbl' or 'lbb'
               need_flatten = True,
               completeness = False):
    if len(image.shape) == 2:
        h,w = image.shape
        ch = 1
        N = h-b+1
        M = w-b+1
        r = np.arange(0,N,s)
        if completeness:
            r = np.hstack((r,np.arange(r[-1]+1,N)))
        c = np.arange(0,M,s)
        if completeness:
            c = np.hstack((c,np.arange(c[-1]+1,M)))
        L = r.shape[0]*c.shape[0]
        if channel_order == 'lbb':
            Px = np.zeros((L,b*b))
        elif channel_order == 'bbl':
            Px = np.zeros((b*b,L))
        else:
            raise AttributeError("The channel_order is not in the choices.")
        k = 0
        for i in range(b):
            for j in range(b):
                blk = image[r+i,:]
                blk = blk[:,c+j]
                li = [blk[:,:] for k in range(ch)]
                flatten_blk = np.vstack(li).reshape((1,-1),order = 'F')
                flatten_blk = np.squeeze(flatten_blk)
                if channel_order == 'bbl':
                    Px[k,:] = flatten_blk
                elif channel_order == 'lbb':
                    Px[:,k] = flatten_blk
                k = k+1
        if not need_flatten:
            if channel_order == 'lbb':
                Px = Px.reshape((L,b,b))
            elif channel_order == 'bbl':
                Px = Px.reshape((b,b,L))
    elif len(image.shape) == 3:
        h,w,ch = image.shape
        N = h-b+1
        M = w-b+1
        r = np.arange(0,N,s)
        if completeness:
            r = np.hstack((r,np.arange(r[-1]+1,N)))
        c = np.arange(0,M,s)
        if completeness:
            c = np.hstack((c,np.arange(c[-1]+1,M)))
        L = r.shape[0]*c.shape[0]
        Px = np.zeros((L,b*b,3))
        k = 0
        for i in range(b):
            for j in range(b):
                blk = image[r+i,:,:]
                blk = blk[:,c+j,:]
                li = [blk[:,:,k] for k in range(ch)]
                flatten_blk = np.hstack(li).reshape((-1,1,3),order = 'F')
                #flatten_blk = np.squeeze(flatten_blk)
                Px[:,k,:] = np.squeeze(flatten_blk)
                k = k+1
        if not need_flatten:
            Px = Px.reshape((L,b,b,ch))
    return Px,ch

"""===============================================================================================
Introduction: The function to show an image by opencv.
---------------------------------------------------------------------------------------------------
Function: Show
Input: image,name = 'Image Show'
		image			----(array) inputed image which must be with type uint8
		name			----(string) the name of Dialog
Return: None
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def Show(image,name = 'Image Show'):
    image = np.uint8(image)
    cv.namedWindow(name)
    cv.imshow(name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""===============================================================================================
Introduction: The function to generate a blur kernel. (Same as fspecial in MATLAB)
---------------------------------------------------------------------------------------------------
Function: fspecial
Input: func_name = 'gaussian',kernel_size=3,sigma=1
		func_name			----(string) The method to generate random mtx. Default input is gaussian.
                                         The orther method hasn't developed yet.
		kernel_size			----(int) The size of kernel
		sigma				----(int) The standard deviation
Return: h                   ----(numpy) The Gaussian Kernel.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
================================================================================================"""
def fspecial(func_name = 'gaussian',kernel_size=3,sigma=1):
    if func_name=='gaussian':
        m = n = (kernel_size-1.)/2.
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h

"""===============================================================================================
Introduction: The function to make convolution on RGB image.
---------------------------------------------------------------------------------------------------
Function: RGB
Input: rgb_mat,g_filter,flag=255
		rgb_mat			----(numpy) inputed image
		g_filter		----(int) kernel
		flag			----(int) threshold
Return:                 ----(float) The convolutional result.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def RGB(rgb_mat,g_filter,flag=255):
    def foo(A,B):
        t=sum(A*B)
        if t>flag: return flag
        return t
    return [foo(rgb_mat[:,:,i],g_filter) for i in range(3)]

"""===============================================================================================
Introduction: The function to make convolution on GRAY image.
---------------------------------------------------------------------------------------------------
Function: GRAY
Input: mat,g_filter,flag = 255
		mat 			----(numpy) inputed image
		g_filter		----(int) kernel
		flag			----(int) threshold
Return:                 ----(float) The convolutional result.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
================================================================================================"""
def GRAY(mat,g_filter,flag = 255):
    def foo(A,B):
        t=np.sum(A*B)
        if t>flag: return flag
        return t
    return foo(mat[:,:],g_filter)

"""===============================================================================================
Introduction: The function to implement Gaussian Filtering.
---------------------------------------------------------------------------------------------------
Function: GaussianFilter
Input: im,g_filter,k=3
		im			    ----(numpy) inputed image
		g_filter		----(numpy) Filter kernel
Return: g_im            ----(numpy) Filtered Image.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def GaussianFilter(im,g_filter,Channel = 'RGB'):
    # Read image data
    # im=cv.imread(image_file)
    if Channel == 'RGB':
        Channel = RGB
    elif Channel == 'GRAY':
        Channel = GRAY
    k = g_filter.shape[0]
    m,n=im.shape
    g_im=im.copy()
    print ('Load Image Data Successful!')
    
    # Initial
    if im.max()>1:
        flag=255
    else:
        flag=1
    #sigma=4
    w=k//2
    #g_filter=fspecial('gaussian',k,sigma)
    # print ('Gaussian Kernel is setup.')
    # print ('The Gaussian Filter is processing...')
    for i in range(w,m-w):
        for j in range(w,n-w):
            t=Channel(im[i-w:i+w+1,j-w:j+w+1],g_filter,flag)
            g_im[i,j]=t
    return g_im

'''=========================================================================================
Introduction: The functin to restore the original image from patches.
---------------------------------------------------------------------------------------------------
Function: RestoreImage
Input: X,step,x_h,x_w
		X			----(numpy) inputed Patches
		s			----(int) block step
		x_h			----(int) or_image height
		x_w	        ----(int) or_image weight
Return: re_img      ----(numpy) restored image
---------------------------------------------------------------------------------------------------
Created on Thu Jan 28 16:53:04 2021
@author: 月光下的云海
========================================================================================='''
def RestoreImage(X,s,x_h,x_w):
    bS = int(X.shape[0]**0.5)
    num_blks = int(X.shape[1]**0.5)
    W = np.zeros(shape = (x_h,x_w))
    ww = np.arange(1,1+bS/s).reshape((-1,1))@np.arange(1,1+bS/s).reshape((1,-1))
    ww = np.kron(ww,np.ones((s,s)))
    W[:bS,:bS] = ww
    W[-bS:,:bS] = np.rot90(ww)
    W[:bS,-bS:] = np.flip(ww,axis = 1)
    W[-bS:,-bS:] = np.rot90(ww,2)
    W[:,bS:-bS] = np.kron(W[:,bS-1].reshape((-1,1)),np.ones((1,x_w-2*bS)))
    W[bS:-bS,:] = np.kron(W[bS-1,:].reshape((1,-1)),np.ones((x_h-2*bS,1)))
    re_img = np.zeros((x_h,x_w))
    
    for i in range(num_blks):
        for j in range(num_blks):
            re_img[s*i:s*i+bS,s*j:s*j+bS] += X[:,num_blks*i+j].reshape((bS,bS),order = 'F')
    re_img = re_img/W
    re_img = re_img.T
    return re_img

"""===============================================================================================
Introduction: The function to realize conv2d like tf.nn.conv2d
---------------------------------------------------------------------------------------------------
Function: conv2d
Input: z, K, b, padding=(0, 0), strides=(1, 1)
		z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
		K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
		b: 偏置,形状(D,)
		padding: padding
		trides: 步长
Return: conv_z: (numpy) conved mtx.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def conv2d(z, K, b, padding=(0, 0), strides=(1, 1)):
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z

"""===============================================================================================
Introduction: The functin to generate the degarded image from blur kernel. The idea is quoted from 
W.S. Dong NCSR. (Dong W , Zhang L , Shi G . Nonlocally Centralized Sparse Representation for Image 
Restoration[J]. IEEE Transactions on Image Processing, 2013, 22(4).)
---------------------------------------------------------------------------------------------------
Function: GetBlurMtx
Input: ker,scale,imshape
		ker			    ----(numpy) The Blur Kernel Mtx
		scale			----(int) Scaling Factor.
		imshape			----(int) The image shape.
Return: A               ----(coo_matrix) The degarded mtx H.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def GetBlurMtx(ker,scale,imshape):
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
    from scipy.sparse import coo_matrix
    A = coo_matrix((V,(R,C)),shape = (M,N))
    return A

"""===============================================================================================
Introduction: The functin to get the DCT Mtx.
---------------------------------------------------------------------------------------------------
Function: overDct
Input: raw,column
		raw			    ----(int) raw numbers
		column			----(int) column numbers
Return: A               ----(numpy) DCT Mtx.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def overDct(raw,column):
    PI = np.pi
    MM = raw**0.5;NN = column**0.5;
    A1 = np.matrix([i for i in range(0,int(MM))])
    A2 = np.matrix([i for i in range(0,int(NN))])
    A = ((2/MM)**0.5)*np.cos((PI/(2*MM))*(np.transpose(A1)*A2))
    A[0,:] = A[0,:]/(2**0.5)
    A = np.kron(A,A)
    return np.array(A)

"""===============================================================================================
Introduction: The functin to denoise an image by NL-Means.
---------------------------------------------------------------------------------------------------
Function: NLMeans
Input: blur_image,blk_size = 8,nlm_h = 10,nlm_num = 12
		blur_image			----(numpy) inputed image
		blk_size			----(int) block size
		nlm_h				----(int) factor h in NLM.
		nlm_num	            ----(BOOL) The matched patches.
Return:                     ----(numpy) Denoised Image.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def NLMeans(blur_image,blk_size = 8,nlm_h = 10,nlm_num = 12):
    #10为滤波器强度，12为匹配个数
    return cv.fastNlMeansDenoising(blur_image,
                                   None,
                                   nlm_h,
                                   blk_size,
                                   nlm_num)

"""===============================================================================================
Introduction: The function to process overflow data.
---------------------------------------------------------------------------------------------------
Function: OverflowPro
Input: im
		im			----(numpy) image array.
output: im          ----(numpy) image array.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def OverflowPro(im):
    im[im > 255] = 255
    im[im < 0] = 0
    return im

"""===============================================================================================
Introduction: The function to setup logger.
---------------------------------------------------------------------------------------------------
Function: setup_logger
Input: logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
		logger_name			----(string) logger name
		root			    ----(root) the path to save logs.
		phase		        ----(string) name to save
		level	            ----(CONSTANT) I don't know
        screen              ----(BOOL) whether to show on screen
        tofile              ----(BOOL) whether to save as **.log
output: None
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
"==============================================================================================="""
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        #log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        log_file = os.path.join(root, phase + '.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


HIGHEST_PROTOCOL = -1
PATH = 'CURVE'

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]

def tick():
    _iter[0] += 1

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value
    
def flush():
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    prints = []
    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig('./'+PATH+'/'+name.replace(' ', '_') + '.jpg')

    print ("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

def Bar1(name,x,xlabel = 'x',ylabel = 'y',labelx = 'x'):
    plt.figure()
    plt.figure(figsize=(7,6))
    sup = len(x)
    #plt.plot(range(sup),x,color = 'orange',linewidth = 1,label = labelx,linestyle='solid')
    #supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.bar(range(sup),x,label = labelx)
    #plt.plot([0,sup],[0,sup],color = 'black',linewidth=1,linestyle='-.')
    plt.xlim([0.0,sup])
    plt.ylim([int(min(x))-1,int(max(x))+1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right")
    plt.savefig('./'+PATH+'/'+name.replace(' ', '_') + '.jpg')
    

def Plt2(name,x,y,xlabel = 'x',ylabel = 'y',labelx = 'x',labely = 'y'):
    plt.figure()
    plt.figure(figsize=(7,6))
    sup = max(len(x),len(y))
    plt.plot(range(len(x)),x,color = 'orange',linewidth = 1,label = labelx)
    plt.plot(range(len(y)),y,color = 'green', linewidth = 1,label = labely)
    plt.plot([0,sup],[0,sup],color = 'black',linewidth=1,linestyle='-.')
    plt.xlim([0.0,sup])
    plt.ylim([int(min(min(x),min(y)))-1,int(max(max(x),max(y)))+1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right")
    plt.savefig('./'+PATH+'/'+name.replace(' ', '_') + '.jpg')

"""===============================================================================================
Introduction: The function to degrad image by filted.
---------------------------------------------------------------------------------------------------
Function: DegradeFilter
Input: hr_image,blur_kernel = fspecial()
		hr_image			----(numpy) hr image matrix.
		blur_kernel			----(numpy) the blur kernel matrix.
output: degraded image.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
================================================================================================"""
def DegradeFilter(hr_image,blur_kernel = fspecial()):
    return cv.filter2D(hr_image.astype(np.uint8),-1,blur_kernel)

"""===============================================================================================
Introduction: The function to convert into image into ycbcr.
---------------------------------------------------------------------------------------------------
Function: rgb2ycbcr
Input: rgb_image
		rgb_image			----(numpy) RGB image matrix.
output:                     ----(numpy) The converted YCbCr image.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
================================================================================================"""
def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""
    if len(rgb_image.shape)!=3 or rgb_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix       
    return ycbcr_image

"""===============================================================================================
Introduction: The function to convert YCbCr image into image into RGB.
---------------------------------------------------------------------------------------------------
Function: ycbcr2rgb
Input: ycbcr_image
		ycbcr_image			----(numpy) YCbCr image matrix.
output:                     ----(numpy) The converted RGB image.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
================================================================================================"""
def ycbcr2rgb(ycbcr_image):
    """convert ycbcr into rgb"""
    if len(ycbcr_image.shape)!=3 or ycbcr_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    ycbcr_image = ycbcr_image.astype(np.float32)
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    shift_matrix = np.array([16, 128, 128])
    rgb_image = np.zeros(shape=ycbcr_image.shape)
    w, h, _ = ycbcr_image.shape
    for i in range(w):
        for j in range(h):
            rgb_image[i, j, :] = np.dot(transform_matrix_inv, ycbcr_image[i, j, :]) - np.dot(transform_matrix_inv, shift_matrix)
    return rgb_image#.astype(np.uint8)

"""===============================================================================================
Introduction: The function to degrad image by bicubic.
---------------------------------------------------------------------------------------------------
Function: DegradeBic
Input: image,scale,is_restore = True
		image			    ----(numpy) RGB image matrix.
        scale               ----(int) The scaling factor.
        is_restore          ----(BOOL) True or False. True for restore image. False for unrestore.
output: lr_img              ----(numpy) The degared image.
---------------------------------------------------------------------------------------------------
Created on Sat Nov 21 10:54:59 2020
@author: 月光下的云海
================================================================================================"""
def DegradeBic(image,scale,is_restore = True):
    h,w = image.shape[0],image.shape[1]
    lr_img = cv.resize(image,(w//scale,h//scale),cv.INTER_CUBIC)
    if is_restore:lr_img = cv.resize(lr_img,(w,h),cv.INTER_CUBIC)
    return lr_img

