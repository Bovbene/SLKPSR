# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:16:43 2021

@author: 月光下的云海
"""
#from PIL import Image
import numpy as np
import _pickle as pickle

class SRBase(object):
    def __init__(self):
        pass

    def upscale(self, im_l, s):
        """
        % im_l: LR image, float np array in [0, 255]
        % im_h: HR image, float np array in [0, 255]
        """
        im_l = im_l/255.0
        if len(im_l.shape)==3 and im_l.shape[2]==3:
            im_l_ycbcr = rgb2ycbcr(im_l)
        else:
            im_l_ycbcr = np.zeros([im_l.shape[0], im_l.shape[1], 3])
            im_l_ycbcr[:, :, 0] = im_l
            im_l_ycbcr[:, :, 1] = im_l
            im_l_ycbcr[:, :, 2] = im_l

        im_l_y = im_l_ycbcr[:, :, 0]*255 #[16 235]
        im_h_y = self.upscale_alg(im_l_y, s)

        # recover color
        if len(im_l.shape)==3:
            im_ycbcr = imresize(im_l_ycbcr, s);
            im_ycbcr[:, :, 0] = im_h_y/255.0; #[16/255 235/255]
            im_h = ycbcr2rgb(im_ycbcr)*255.0
        else:
            im_h = im_h_y

        im_h = np.clip(im_h, 0, 255)
        im_h_y = np.clip(im_h_y, 0, 255)
        return im_h,im_h_y

    def upscale_alg(self, im_l_y, s):
        pass

class Bicubic(SRBase):
    def upscale_alg(self, im_l_y, s):
        im_h_y = imresize(im_l_y, s)
        return im_h_y

class SCN(SRBase):
    def __init__(self, model_files):
        self.mdls = []
        for f in model_files:
            self.mdls += [pickle.load(open(f, 'rb'),encoding = 'bytes')]
        i=model_files[0].find('_x')
        self.MDL_SCALE = int(model_files[0][i+2]);
        self.PATCH_SIZE = 5
        self.BORDER_SIZE = 6
        self.SCALE_Y = 1.1 #linear factor on scaley layer

    def upscale_alg(self, im_l_y, s):
        h_gt, w_gt = im_l_y.shape[0]*s, im_l_y.shape[1]*s
        hpsz = self.PATCH_SIZE/2

        itr_all = int(np.ceil(np.log(s)/np.log(self.MDL_SCALE)))
        for itr in range(itr_all):
            print('itr:', itr)
            im_y = imresize(im_l_y, self.MDL_SCALE)
            im_y = ExtendBorder(im_y, self.BORDER_SIZE)
            mdl=self.mdls[itr]

            # extract gradient features
            convfea = ExtrConvFea(im_y, mdl[b'conv'])
            im_mean = ExtrConvFea(im_y, mdl[b'mean2'])
            diffms = ExtrConvFea(im_y, mdl[b'diffms'])

            # matrix operation
            h, w, c = convfea.shape
            convfea = convfea.reshape([h*w, c])
            convfea_norm = np.linalg.norm(convfea, axis=1)
            convfea = (convfea.T/convfea_norm).T
            wd = np.dot(convfea, mdl[b'wd'])
            z0 = ShLU(wd, 1)
            z = ShLU(np.dot(z0, mdl[b'usd1'])+wd, 1) #sparse code

            hPatch = np.dot(z, mdl[b'ud'])
            hNorm = np.linalg.norm(hPatch, axis=1)
            diffms = diffms.reshape([h*w, diffms.shape[2]])
            mNorm = np.linalg.norm(diffms, axis=1)
            hPatch = (hPatch.T/hNorm*mNorm).T*self.SCALE_Y
            hPatch = hPatch*mdl[b'addp'].flatten()

            hPatch = hPatch.reshape([h, w, hPatch.shape[1]])
            im_h_y = im_mean[:, :, 0]
            h, w = im_h_y.shape
            cnt = 0
            for ii in range(self.PATCH_SIZE-1, -1, -1):
                for jj in range(self.PATCH_SIZE-1, -1, -1):
                    im_h_y = im_h_y+hPatch[jj:(jj+h), ii:(ii+w), cnt]
                    cnt = cnt+1
            
            im_l_y = im_h_y

        # shrink size to gt
        if (im_h_y.shape[0]>h_gt):
            print('downscale from {} to {}'.format(im_h_y.shape, (h_gt, w_gt)))
            im_h_y = imresize(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            assert(im_h_y.shape[1]==w_gt)

        return im_h_y
    
import cv2

def modcrop(im, modulo):
    sz = im.shape
    h = sz[0]//modulo*modulo
    w = sz[1]//modulo*modulo
    ims = im[0:h, 0:w, ...]
    return ims

def imresize(im_l, s):
    if s<1:
        im_l = cv2.GaussianBlur(im_l, (7,7), 0.5)
    im_h = cv2.resize(im_l, (0,0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return im_h

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def shave(im, border):
    if isinstance(border, int):
        border=[border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im

def ExtendBorder(im, offset):
    sz = im.shape
    assert(len(sz)==2)

    im2 = np.zeros([sz[0]+offset*2, sz[1]+offset*2])
    im2[ offset:-offset, offset:-offset ] = im
    im2[ offset:-offset, 0:offset ] = im[:, offset:0:-1]
    im2[ offset:-offset, -offset: ] = im[:, -2:-(offset+2):-1]
    im2[ 0:offset, :] = im2[2*offset:offset:-1, :]
    im2[ -offset:, :] = im2[-(offset+2):-(2*offset+2):-1, :]

    return im2

def ExtrConvFea(im, fltrs):
    """
    % extract convoluation features from whole image output
    % fea: [mxnxf], where f is the number of features used
    """
    m,n = im.shape
    nf = fltrs.shape[1]
    fs = int(np.round(np.sqrt(fltrs.shape[0])))
    hfs = fs//2
    fea = np.zeros([m-fs+1, n-fs+1, nf])
    for i in range(nf):
        fltr = fltrs[:, i].reshape([fs, fs])
        acts = cv2.filter2D(im, -1, fltr)
        fea[:, :, i] = acts[hfs:-hfs, hfs:-hfs]
    return fea

def ShLU(a, th):
    return np.sign(a)*np.maximum(0, np.abs(a)-th)
    