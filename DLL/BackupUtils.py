# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:10:40 2021

@author: 月光下的云海
"""
from DLL.SparseCoding import shrink
import cv2 as cv
import numpy as np
from DLL.utils import GetAbsMax

def kernel_init(shape,stddev = 5e-2):
    init = tf.truncated_normal_initializer(stddev = 5e-2)
    return(tf.get_variable(shape = shape,initializer = init,name = 'weight'))

def bias_init(shape):
    init = tf.constant_initializer(0.0)
    return(tf.get_variable(shape = shape,initializer = init,name = 'bias'))

def conv2d(x,kernel,
            bias,
            strides,
            padding = 'SAME',
            act = tf.nn.relu):
    #in_depth = x.get_shape().as_list()[-1]
    #with tf.variable_scope(scope,reuse = reuse):
    #shape = ksize+[in_depth,out_depth]
    strides = [1,strides,strides,1]
    conv = tf.nn.conv2d(x,kernel,strides,padding,name = 'conv')
    # with tf.variable_scope('bias'):
    #     bias = bias_init([out_depth])
    out = tf.nn.bias_add(conv,bias)
    out = act(out)
    return out

def Test3(self,image = None,image_path = None):
    if (image is None) and (image_path is None):
        raise AttributeError("U must input an image path or an image mtx.")
    elif (image is None) and (image_path is not None):
        lr_image = cv.imread(image_path)
    elif (image is not None) and (image_path is None):
        lr_image = image
    else:
        raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
    if self.is_norm:
        lr_image = lr_image/255
    self.batch_size = 1
    input_ph = tf.placeholder(shape = (1,)+lr_image.shape,dtype = tf.float32,name = 'TestInputPh')
    if self.blur_kernel is not None: 
        self.SolveH(int(input_ph.shape[1]),int(input_ph.shape[2]))
        self.H = self.H.toarray()
    self.H = tf.constant(name = 'H',dtype=tf.float32,value=self.H)
    output = self.EntireSR(input_ph,self.H)
    del self.H
    saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())
    saver.restore(self.sess, self.save_path+'model.ckpt')
    t = time()
    result = self.sess.run(output,feed_dict = {input_ph:lr_image.reshape((1,)+lr_image.shape)})
    result = OverflowPro(result)
    print("Time Elapsed:",time()-t)
    if self.is_norm:
        result = result*255
    
    if image_path is not None:
        cv.imwrite(self.test_save_path+'x'+str(self.scale)+split(image_path)[-1],np.squeeze(result))
    return np.squeeze(result)

def Test(self,image = None, image_path = None):
    if (image is None) and (image_path is None):
        raise AttributeError("U must input an image path or an image mtx.")
    elif (image is None) and (image_path is not None):
        lr_image = cv.imread(image_path)
    elif (image is not None) and (image_path is None):
        lr_image = image
    else:
        raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
    if self.is_norm:
        lr_image = lr_image/255
    self.batch_size = 1
    
    delta = 7
    
    blur_kernel = fspecial(kernel_size = 7,sigma = 4)
    in_im = tf.placeholder(shape = (1,)+lr_image.shape,dtype = tf.float32,name = 'TestInputPh')
    
    l,lr_h,lr_w,ch = in_im.get_shape().as_list()
    hr_h,hr_w = lr_h*self.scale,lr_w*self.scale
    
    sr_in = tf.placeholder(shape = [l,hr_h,hr_w,ch],dtype = tf.float32,name = 'SRCNNIN')
    alphaX = tf.placeholder(shape = [l,hr_h*hr_w,128],dtype = tf.float32,name = 'DecodeIn')
    
    y_srcnn = self.SRCNN(sr_in)
    alpha,im_h,im_w = self.SparseEncode(sr_in)
    decode_im = self.DictionaryLearningDecode(alphaX, hr_h, hr_w)
    
    hr_im = cv.resize(lr_image,(hr_h,hr_w),cv.INTER_CUBIC)
    
    
    HR = GetBlurMtx(blur_kernel,self.scale,imshape = (int(in_im.shape[1]),int(in_im.shape[2])))
    HG = GetBlurMtx(blur_kernel,self.scale,imshape = (int(in_im.shape[1]),int(in_im.shape[2])))
    HB = GetBlurMtx(blur_kernel,self.scale,imshape = (int(in_im.shape[1]),int(in_im.shape[2])))
    
    y = lr_image.reshape((-1,3))
    HRTY = HR.T @ y[:,0]
    HRTHR = HR.T@ HR
    HGTY = HG.T @ y[:,1]
    HGTHG = HG.T @ HG
    HBTY = HB.T @ y[:,2]
    HBTHB = HB.T @ HB
    print(HRTY.shape,HRTHR.shape)
    for ite in range(4):
        f = np.squeeze(hr_im)
        X_m = self.sess.run(y_srcnn,feed_dict = {sr_in:hr_im.reshape((1,hr_h,hr_w,ch))})
        for k in range(4):
            f = f.reshape( (hr_h*hr_w,ch) )
            rf = np.array(f[:,0],dtype = np.float)
            gf = np.array(f[:,1],dtype = np.float)
            bf = np.array(f[:,2],dtype = np.float)
            print(rf.shape)
            for _ in range(5):
                rf += delta * (HRTY-HRTHR @ rf)
                gf += delta * (HGTY-HGTHG @ gf)
                bf += delta * (HBTY-HBTHB @ bf)
            rf = rf.reshape((hr_h,hr_w,1),order = 'F')
            gf = gf.reshape((hr_h,hr_w,1),order = 'F')
            bf = bf.reshape((hr_h,hr_w,1),order = 'F')
            f = np.concatenate([rf,gf,bf],axis = 2)
            if k % 4 == 0:
                X_m = self.sess.run(y_srcnn,feed_dict = {sr_in:f.reshape((1,)+f.shape)})
            X = f.reshape((self.batch_size,hr_h,hr_w,self.ch))
            alpha_X = self.sess.run(alpha,feed_dict = {sr_in:X})
            alpha_X_m = self.sess.run(alpha,feed_dict = {sr_in:X_m})
            alpha_X = alpha_X-alpha_X_m
            hr_im = self.sess.run(decode_im,feed_dict = {alphaX:alpha_X})+X_m
    return OverflowPro(hr_im)
                



def Train2(self,x_data,y_data):
    if not exists(self.save_path):
        makedirs(self.save_path)
    setup_logger('base','./TRAINED_MODEL/','train_on_'+self.train_name, level=logging.INFO,screen=True, tofile=True)
    logger = logging.getLogger('base')
    hr_h,hr_w = self.blk_size*self.scale,self.blk_size*self.scale
    self.in_im = tf.placeholder(shape = [self.batch_size,self.blk_size,self.blk_size,self.ch],dtype = tf.float32,name = 'InputImagePh')
    self.label_im = tf.placeholder(shape = [self.batch_size,hr_h,hr_w,self.ch],dtype = tf.float32,name = 'LableImagePh')
    
    flag1 = tf.placeholder(tf.float32, shape=[], name='Flag1')
    flag2 = tf.placeholder(tf.float32, shape=[], name='Flag2')
    flag3 = tf.placeholder(tf.float32, shape=[], name='Flag3')
    
    if self.blur_kernel is not None: 
        self.SolveH(int(self.in_im.shape[1]),int(self.in_im.shape[2]))
        self.H = self.H.toarray()
    self.H = tf.constant(name = 'H',dtype=tf.float32,value=self.H)
    
    bic_im = tf.image.resize_bicubic(self.in_im, (hr_h,hr_w))
    srop_im = self.SRCNN(bic_im)
    psnr1 = tf.image.psnr(self.label_im, srop_im, max_val=255)
    ssim1 = tf.image.ssim(self.label_im, srop_im, max_val=255)
    
    alpha,im_h,im_w = self.SparseEncode(self.label_im)
    decode_im = self.DictionaryLearningDecode(alpha, im_h, im_w)
    
    all_vars = tf.global_variables()
    sr_vars = [v for v in all_vars if 'SRCNN' in v.name]
    ed_vars = [v for v in all_vars if ('SparseEncode' in v.name) or ('DictionaryLearningDecode' in v.name)]
    
    # opt_srop,srop_loss = self.GenerateTrainOp(self.label_im,srop_im,var_list = sr_vars,loss_function = l1_loss)
    # opt_ed,ed_loss = self.GenerateTrainOp(self.label_im,decode_im,var_list = ed_vars,loss_function = l2_loss)
    srop_loss = l1_loss(self.label_im,srop_im)
    ed_loss = l2_loss(self.label_im,decode_im)
    
    # self.sr_im = self.EntireSR(self.in_im,self.H)
    self.sr_im = self.EntireSR(self.in_im)
    
    ent_loss = flag1*l1_loss(self.label_im, self.sr_im) + flag2*ed_loss + flag3*srop_loss
    opt_srop = tf.train.AdamOptimizer(self.lr).minimize(ent_loss, var_list=sr_vars)
    opt_ed = tf.train.AdamOptimizer(self.lr).minimize(ent_loss, var_list=ed_vars)
    #opt_ent = tf.train.AdamOptimizer(self.lr).minimize(ent_loss)
    
    # ent_loss = l1_loss(self.label_im, self.sr_im)
    # opt_ent = tf.train.AdamOptimizer(self.lr).minimize(ent_loss)
    #opt_ent, ent_loss = self.GenerateTrainOp(self.label_im,self.sr_im,var_list = None,loss_function = l1_loss)
    psnr2 = tf.image.psnr(self.label_im, self.sr_im, max_val=255)
    ssim2 = tf.image.ssim(self.label_im, self.sr_im, max_val=255)
    saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())
    logger.info('Pretraining.....................')
    flag1_ = 0
    train_srop_psnr_li = []
    train_srop_ssim_li = []
    train_ent_psnr_li = []
    train_ent_ssim_li = []
    for epoch in range(self.pre_train_epoch):
        xtrain_li,ytrain_li,xtest_li,ytest_li = self.GetDataGroup(x_data,y_data)
        start_t = time()
        for x_train,y_train in zip(xtrain_li,ytrain_li):
            flag2_,flag3_ = 0,1
            feed_dict = {self.in_im:x_train,self.label_im:y_train,flag1:flag1_,flag2:flag2_,flag3:flag3_}
            train_srop_psnr,train_srop_ssim,_ = self.sess.run([psnr1,ssim1,opt_srop],feed_dict = feed_dict)
            flag2_,flag3_ = 1,0
            feed_dict = {self.in_im:x_train,self.label_im:y_train,flag1:flag1_,flag2:flag2_,flag3:flag3_}
            train_ed_loss,_ = self.sess.run([ed_loss,opt_ed],feed_dict = feed_dict)
        test_idx = np.random.randint(0,len(xtest_li),size = 1)[0]
        x_test,y_test = xtest_li[test_idx],ytest_li[test_idx]
        test_srop_psnr,test_srop_ssim = self.sess.run([psnr1,ssim1],feed_dict = {self.in_im:x_test,self.label_im:y_test})
        test_ed_loss = self.sess.run(ed_loss,feed_dict = {self.label_im:y_test})
        test_ent_psnr,test_ent_ssim = self.sess.run([psnr2,ssim2],feed_dict = {self.in_im:x_test,self.label_im:y_test})
        logger.info( ("[%3f][Epocch:%2d/%2d], SROP_PSNR(Train): %5f, SROP_SSIM(Train): %5f, SROP_PSNR(Test): %5f, SROP_SSIM(Test):%5f, ED_Loss:%5f, EntireSR_PSNR:%5f, EntireSR_SSIM:%5f."
                      %(time()-start_t,epoch+1,self.pre_train_epoch,ave(train_srop_psnr),ave(train_srop_ssim),ave(test_srop_psnr),ave(test_srop_ssim),test_ed_loss,ave(test_ent_psnr),ave(test_ent_ssim)) ) )
        train_srop_psnr_li += [ave(train_srop_psnr)]
        train_srop_ssim_li += [ave(train_srop_ssim)]
        train_ent_psnr_li += [ave(test_ent_psnr)]
        train_ent_ssim_li += [ave(test_ent_ssim)]
        del xtrain_li,ytrain_li,xtest_li,ytest_li
        # print('Epoch:{}, Train PSNR of SROP:{:.5f}, Train SSIM of SROP{:.5f}, Test PSNR of SROP:{:.5f}, Test SSIM of SROP:{:.5f}, Test Loss of ED:{:.5f}'
        #       .format(epoch+1,ave(train_srop_psnr),ave(train_srop_ssim),ave(test_srop_psnr),ave(test_srop_ssim),test_ed_loss))
        # print('  Test PSNR of EntireSR:{:.5f}, Test SSIM of EntireSR:{:.5f}'
        #       .format(ave(test_ent_psnr),ave(test_ent_ssim)))
    
    logger.info('Training.....................')
    flag1_,flag2_,flag3_ = 1,0,0
    Threshold = -np.Inf
    for epoch in range(self.pre_train_epoch,self.epochs):
        start_t = time()
        xtrain_li,ytrain_li,xtest_li,ytest_li = self.GetDataGroup(x_data,y_data)
        for x_train,y_train in zip(xtrain_li,ytrain_li):
            feed_dict = {self.in_im:x_train,self.label_im:y_train,flag1:flag1_,flag2:flag2_,flag3:flag3_}
            self.sess.run(opt_ed,feed_dict = feed_dict)
            feed_dict = {self.in_im:x_train,self.label_im:y_train,flag1:flag1_,flag2:flag2_,flag3:flag3_}
            train_ent_psnr,train_ent_ssim,_ = self.sess.run([psnr2,ssim2,opt_srop],feed_dict = feed_dict)
            #feed_dict = {self.in_im:x_train,self.label_im:y_train,flag1:flag1_,flag2:flag2_,flag3:flag3_}
            #train_ent_psnr,train_ent_ssim,_ = self.sess.run([psnr2,ssim2,opt_ent],feed_dict = feed_dict)
        if ave(train_ent_psnr) > Threshold:
            logger.info('Saving the better model......')
            saver.save(sess = self.sess,save_path = self.save_path+'model.ckpt')
            Threshold = ave(train_ent_psnr)
            ThSSIM = ave(train_ent_ssim)
        test_idx = np.random.randint(0,len(xtest_li),size = 1)[0]
        x_test,y_test = xtest_li[test_idx],ytest_li[test_idx]
        test_ent_psnr,test_ent_ssim = self.sess.run([psnr2,ssim2],feed_dict = {self.in_im:x_test,self.label_im:y_test})
        logger.info( ("[%3f][Epocch:%2d/%2d], EntireSR_PSNR(Train):%5f, EntireSR_SSIM(Train)):%5f, Entire+SR_PSNR(Test):%5f, EntireSR_SSIM(Test):%5f."%
                      (time()-start_t,epoch+1,self.epochs,ave(train_ent_psnr),ave(train_ent_ssim),ave(test_ent_psnr),ave(test_ent_ssim))) )
        # print('Epoch:{}, Train PSNR of EntireSR:{:.5f}, Train SSIM of EntireSR{:.5f}, Test PSNR of EntireSR:{:.5f}, Test SSIM of EntireSR:{:.5f}'
        #       .format(epoch+1,ave(train_ent_psnr),ave(train_ent_ssim),ave(test_ent_psnr),ave(test_ent_ssim)))
        train_ent_psnr_li += [ave(train_ent_psnr)]
        train_ent_ssim_li += [ave(train_ent_ssim)]
    Plt2('PSNR curve',train_ent_psnr_li,train_srop_psnr_li,xlabel = 'Epoches',ylabel = 'PSNR Value',labelx = 'SLBPSR',labely = 'SRCNN')
    Plt2('SSIM curve',train_ent_ssim_li,train_srop_ssim_li,xlabel = 'Epoches',ylabel = 'SSIM Value',labelx = 'SLBPSR',labely = 'SRCNN')
    logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}),(SSIM:{:.4f}).'.format(Threshold, ThSSIM))
    # info = '\nInput Placeholder:'+str(self.in_im.name)+'\nOutput Tensor:'+str(self.EntireSR.name)+'\nLoss Tensor:'+str(ent_loss.name)+'\nOutput Placeholder:'+str(self.label_im.name)
    # logger.info(info)
    self.sess.close()


def EntireSR(self,in_im,H = None):
    l,lr_h,lr_w,ch = in_im.get_shape().as_list()
    hr_h,hr_w = lr_h*self.scale,lr_w*self.scale
    #lr_h,lr_w = self.blk_size,self.blk_size
    hr_im = tf.image.resize_bicubic(in_im, (hr_h,hr_w))
    #l,im_h,im_w,ch = hr_im.get_shape().as_list()
    
    stretch = tf.reshape(in_im,(self.batch_size,lr_h*lr_w,self.ch))
    
    if len(H.get_shape().as_list()) == 2:
        HT = tf.stack([tf.transpose(H) for _ in range(3)],axis = 2)
        H = tf.stack([H for _ in range(3)],axis = 2)
    
    HTH = Matmul3D(HT, H)
    HTY = Matmul3D(stretch,H)
    
    for ite in range(4):
        f = hr_im
        X_m = self.SRCNN(hr_im,reuse = True)
        for k in range(4):
            
            f = tf.reshape(f,(self.batch_size,hr_h*hr_w,self.ch))
            for _ in range(5):
                f += 7*(HTY-Matmul3D(f,HTH))
            if k % 4 == 0:
                X_m = self.SRCNN(tf.reshape(f,(self.batch_size,hr_h,hr_w,self.ch)),reuse = True)
            X = tf.reshape(f,(self.batch_size,hr_h,hr_w,self.ch))
            alpha_X,_,_ = self.SparseEncode(X,reuse = True)
            alpha_X_m,_,_ = self.SparseEncode(X_m,reuse = True)
            alpha_X = alpha_X-alpha_X_m
            hr_im = self.DictionaryLearningDecode(alpha_X, hr_h, hr_w,reuse = True)+X_m
            
    return hr_im


def SparseEncode(self,Iy,scope = 'SparseEncode',reuse = None):
    with tf.variable_scope(scope,reuse = reuse):
        _,im_h,im_w,_ = Iy.get_shape().as_list()
        y = conv2d(Iy,100,9,stride = 1,activation_fn = None,padding = 'SAME',scope = 'ConvH')
        y = tf.reshape(y,(self.batch_size,-1,100))
        W = self.GetWeight([100,128],name = 'W')
        S = self.GetWeight([128,128],name = 'S')
        y = tf.matmul(y,W)
        alpha = y
        for i in range(self.max_iter):
            alpha = self.Shrinkage(alpha,name = 'Theta{}'.format(i))
            alpha = tf.matmul(alpha,S)
            alpha = y+alpha
        alpha = self.Shrinkage(alpha,name = 'ThetaEnd')
        return alpha,im_h,im_w

def Shrinkage(self,x,name):
    theta = self.GetWeight([1,128],name = name)
    alpha = tf.div(x,theta)
    alpha = self.Shrink(alpha,1)
    alpha = tf.multiply(alpha,theta)
    return alpha

def SRCNN(self,x,scope = 'SRCNN',reuse = None):
    with tf.variable_scope(scope,reuse = reuse):
        with slim.arg_scope([slim.conv2d],activation_fn = self.lrelu):
            y = slim.conv2d(x,64,3,stride = 1,padding = 'SAME',scope = 'conv1')
            y = slim.conv2d(y,32,1,stride = 1,padding = 'SAME',scope = 'conv2')
            y = slim.conv2d(y, self.ch,5,stride = 1,padding = 'SAME',scope = 'conv3',activation_fn = None)
            return y


def DictionaryLearningDecode(self,alpha,im_h,im_w,scope = 'DictionaryLearningDecode',reuse = None):
    with tf.variable_scope(scope,reuse = reuse):
        Dx = self.GetWeight([128,25],name = 'Dx')
        x = tf.matmul(alpha,Dx)
        x = tf.reshape(x,(-1,im_h,im_w,25))
        Ix = conv2d(x,self.ch,5,stride = 1,activation_fn = None,padding = 'SAME',scope = 'ConvG')
        return Ix 


def EntireSR(self,in_im):
    _,lr_h,lr_w,_ = in_im.get_shape().as_list()
    hr_h,hr_w = lr_h*self.scale,lr_w*self.scale
    #lr_h,lr_w = self.blk_size,self.blk_size
    bic_im = tf.image.resize_bicubic(in_im, (hr_h,hr_w))
    hr_im = tf.image.resize_bicubic(in_im, (hr_h,hr_w))
    #l,im_h,im_w,ch = hr_im.get_shape().as_list()
    
    #stretch = tf.reshape(in_im,(self.batch_size,lr_h*lr_w,self.ch))
    
    # if len(H.get_shape().as_list()) == 2:
    #     HT = tf.stack([tf.transpose(H) for _ in range(3)],axis = 2)
    #     H = tf.stack([H for _ in range(3)],axis = 2)
    
    # HTH = Matmul3D(HT, H)
    # HTY = Matmul3D(stretch,H)
    HTY = self.Hf(self.HT,bic_im)
    for ite in range(4):
        f = hr_im
        X_m = self.SRCNN(hr_im)
        for k in range(4):
            
            # f = conv2d(f-X_m,self.EntireSR_Conv1,self.EntireSR_Bais1,strides = 1,padding = 'SAME',act = tf.identity)
            # f += HTY-self.Hf(self.HT,self.Hf(self.H,hr_im))
            # f = f-conv2d(hr_im,self.EntireSR_Conv2,self.EntireSR_Bais2,strides = 1,padding = 'SAME',act = tf.identity)
            
            #f = tf.reshape(f,(self.batch_size,hr_h*hr_w,self.ch))
            
            #f = HTY-self.Hf(self.HT,self.Hf(self.H,f))
            #f += 7*(HTY-self.Hf(self.HT,self.Hf(self.H,f)))
            
            # for _ in range(5):
            #     f += 7*(HTY-self.Hf(self.HT,self.Hf(self.H,f)))
            # f = HTY-Matmul3D(f,HTH)
            # for _ in range(5):
            #     f += 7*(HTY-Matmul3D(f,HTH))
            # if k % 4 == 0:
            #     X_m = self.SRCNN(tf.reshape(f,(self.batch_size,hr_h,hr_w,self.ch)) )
            X = tf.reshape(f,(self.batch_size,hr_h,hr_w,self.ch))
            alpha_X,_,_ = self.SparseEncode(X)
            alpha_X_m,_,_ = self.SparseEncode(X_m)
            alpha_X = alpha_X-alpha_X_m
            hr_im = self.DictionaryLearningDecode(alpha_X, hr_h, hr_w)+X_m
            
    return hr_im


def CalTheta(para,answer):
    theta = np.random.normal(size = para[0].shape,loc = 10) 
    length = len(answer)
    for i, (parai,answeri) in enumerate(zip(para,answer)):
        norm_factor = min(GetAbsMax(parai),GetAbsMax(answeri))
        para[i] = parai/norm_factor
        answer[i] = answeri/norm_factor
        answer[i] = answer[i]-theta[length:].T @ para[i][4:]
    answer = np.array(answer).reshape((-1,1))
    para = [parai[:length].T for parai in para]
    para = np.vstack(para)
    theta[:length] = np.linalg.inv(para) @ answer
    return theta

def SolveHR(b,DH,blur_kernel,blk_size,apha_l = None,hr = None,eps = 1):

    #初始化b，alpha_l和高分字典
    #b = blur_bloc_im_li[0][0].row_stretch
    #apha_l = blur_bloc_im_li[0][0].sparse_code
    if apha_l is None:
        apha_l = np.ones(b.shape)
    #hr = hr_bloc_im_li[0][0].row_stretch
    #DH = hr_dic_li[0].dic
    #eps = 100
    for _ in range(10):
        #计算bTAa
        bH = DH @ apha_l
        if hr is not None:
            res = np.linalg.norm((DH @ apha_l).reshape((-1))-hr)
        #bH.reshape((blk_size,blk_size),order = 'F')
        KbH = cv.filter2D(bH.reshape((blk_size,blk_size),order = 'F'),-1,blur_kernel)
        KbH = KbH.reshape((-1,1),order = 'F')
        bTAa = np.squeeze(b.T @ KbH)
        #计算Aa的二范数
        Aa2 = np.linalg.norm(KbH,ord = 2)
        #计算Theta
        theta = CalTheta(para1 = b*apha_l,answer1 = bTAa,para2 = b,answer2 = Aa2)
        Theta = np.diag(theta.reshape((-1)))
        
        #c = np.abs(theta).max()+eps
        c = 1
        apha_l = shrink(1/c * Theta@(b-KbH)+apha_l)
        if hr is not None and np.linalg.norm((DH @ apha_l).reshape((-1))-hr) > res:
            break
        elif hr is not None:
            print(np.linalg.norm((DH @ apha_l).reshape((-1))-hr))
        else:
            continue
    return apha_l,bH

def SolveHB(b,DH,blur_kernel,blk_size,apha_l,hr = None):
    alpha = apha_l
    eps = 1
    while True:
        bH = DH @ alpha
        res = np.linalg.norm(bH.reshape((-1))-hr)
        KbH = cv.filter2D(bH.reshape((blk_size,blk_size),order = 'F'),-1,blur_kernel)
        KbH = KbH.reshape((-1,1),order = 'F')
        bTAa = float(b.T @ KbH)
        Aa2 = np.linalg.norm(KbH,ord = 2)**2
        theta = CalTheta(para1 = b*alpha,answer1 = bTAa,para2 = alpha*bH,answer2 = Aa2)
        Theta = np.diag(theta.reshape((-1)))
        c = np.abs(theta).max()**2+eps
        alpha = shrink(1/c * Theta@(b-KbH)+alpha)
        if (hr is not None) and (np.linalg.norm((DH @ alpha).reshape((-1))-hr) > res):
            print('-------------')
            break
        elif (hr is not None) and (np.linalg.norm((DH @ alpha).reshape((-1))-hr) <= res):
            print(np.linalg.norm((DH @ alpha).reshape((-1))-hr))
        else:
            continue
    return alpha,bH

def Test2(self,image = None,image_path = None):
        
    if (image is None) and (image_path is None):
        raise AttributeError("U must input an image path or an image mtx.")
    elif (image is None) and (image_path is not None):
        lr_image = cv.imread(image_path)
    elif (image is not None) and (image_path is None):
        lr_image = image
    else:
        raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
    if self.is_norm:
        lr_image = lr_image/255
    
    #input_ph = tf.placeholder(shape = (1,)+lr_image.shape,dtype = tf.float32,name = 'TestInputPh')
    
    Px,_ = GetPatches(lr_image,self.blk_size,self.blk_size//2,need_flatten = False,completeness = True)
    self.batch_size = Px.shape[0]
    input_ph = tf.placeholder(shape = Px.shape,dtype = tf.float32,name = 'TestInputPh')

    if self.blur_kernel is not None: 
        self.SolveH(int(input_ph.shape[1]),int(input_ph.shape[2]))
        self.H = self.H.toarray()
    self.H = tf.constant(name = 'H',dtype=tf.float32,value=self.H)
    
    output = self.EntireSR(input_ph,self.H)
    del self.H
    saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())
    saver.restore(self.sess, self.save_path+'model.ckpt')
    t = time()
    
    result = self.sess.run(output,feed_dict = {input_ph:Px})
    result = OverflowPro(result)
    
    def restore_channel(result):
        mtx = np.transpose(result.reshape((self.batch_size,-1)))
        return RestoreImage(mtx,self.scale*self.blk_size//2,lr_image.shape[0]*self.scale,lr_image.shape[1]*self.scale)
    R = restore_channel(result[:,:,:,0]).reshape((lr_image.shape[0]*self.scale,lr_image.shape[1]*self.scale,1))
    G = restore_channel(result[:,:,:,1]).reshape((lr_image.shape[0]*self.scale,lr_image.shape[1]*self.scale,1))
    B = restore_channel(result[:,:,:,2]).reshape((lr_image.shape[0]*self.scale,lr_image.shape[1]*self.scale,1))
    
    result = np.concatenate([R,G,B],axis = 2)
    
    #result = RestoreImage(np.squeeze(result),self.blk_size*2,lr_image.shape[0]*self.scale,lr_image.shape[1]*self.scale)
    #result = self.sess.run(output,feed_dict = {input_ph:lr_image.reshape((1,)+lr_image.shape)})
    print("Time Elapsed:",time()-t)
    if self.is_norm:
        result = result*255
    
    if image_path is not None:
        cv.imwrite(self.test_save_path+'x'+str(self.scale)+split(image_path)[-1],np.squeeze(result))
    return np.squeeze(result)

def Test3(self,image = None,image_path = None):
    if (image is None) and (image_path is None):
        raise AttributeError("U must input an image path or an image mtx.")
    elif (image is None) and (image_path is not None):
        lr_image = cv.imread(image_path)
    elif (image is not None) and (image_path is None):
        lr_image = image
    else:
        raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
    self.batch_size = 1
    ycbcr = cv.cvtColor(lr_image,cv.COLOR_BGR2YCrCb)
    h,w = lr_image.shape[0],lr_image.shape[1]
    hr_h,hr_w = h*self.scale,w*self.scale
    bic_image = cv.resize(lr_image,(hr_w,hr_h),cv.INTER_CUBIC)
    cb,cr = cv.cvtColor(bic_image,cv.COLOR_BGR2YCrCb)[:,:,1],cv.cvtColor(bic_image,cv.COLOR_BGR2YCrCb)[:,:,2]
    
    lr_image = ycbcr[:,:,0]
    in_im = tf.placeholder(shape = [1,w,h,1],dtype = tf.float32,name = 'TestInputPh')
    if self.blur_kernel is not None: 
        self.SolveH(int(in_im.shape[1]),int(in_im.shape[2]))
    HT = Dense2Sparse(tf.constant(name = 'HT',dtype=tf.float32,value=np.transpose(self.H.toarray())))
    HTH = Dense2Sparse( tf.constant(name = 'HTH',dtype=tf.float32,value=(np.transpose(self.H) @ self.H).toarray()) )
    self.H = Dense2Sparse(tf.constant(name = 'H',dtype=tf.float32,value=self.H.toarray()))
    sr_im = self.EntireSR(in_im,self.H,HT,HTH)
    t = time()
    result = self.sess.run(sr_im,feed_dict = {in_im:lr_image.reshape((1,)+lr_image.shape+(1,))})
    print("Time Elapsed:",time()-t)
    result = np.squeeze(result)
    print(result)
    result = np.stack((result,cb,cr),axis = 2)
    result = cv.cvtColor(np.uint8(result),cv.COLOR_YCrCb2BGR)
    return result

def Test(self,image = None,image_path = None):
    if (image is None) and (image_path is None):
        raise AttributeError("U must input an image path or an image mtx.")
    elif (image is None) and (image_path is not None):
        lr_image = cv.imread(image_path)
    elif (image is not None) and (image_path is None):
        lr_image = image
    else:
        raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
    
    h,w = lr_image.shape[0],lr_image.shape[1]
    hr_h,hr_w = h*self.scale,w*self.scale
    def srch(y):
        # h,w = y.shape[0],y.shape[1]
        # hr_h,hr_w = h*self.scale,w*self.scale
        Py,_ = GetPatches(y,
               self.blk_size,
               self.blk_size//2,
               channel_order = 'lbb',#'bbl' or 'lbb'
               need_flatten = False,
               completeness = True)
        self.batch_size = Py.shape[0]
        in_im = tf.placeholder(shape = Py.shape+(1,),dtype = tf.float32,name = 'TestInputPh')
        if self.blur_kernel is not None: 
            self.SolveH(int(in_im.shape[1]),int(in_im.shape[2]))
            HTH = super(SpConvSR,self).SparseMTX2SparseTensor(self.H.T @ self.H)
            H = super(SpConvSR,self).GetSparseH(self.blur_kernel, self.scale, (int(in_im.shape[1]),int(in_im.shape[2])))
            HT = super(SpConvSR,self).GetSparseH(self.blur_kernel, self.scale, (int(in_im.shape[1]),int(in_im.shape[2])),is_transpose = True)
            
        #HT = Dense2Sparse(tf.constant(name = 'HT',dtype=tf.float32,value=np.transpose(self.H.toarray())))
        #HTH = Dense2Sparse( tf.constant(name = 'HTH',dtype=tf.float32,value=(np.transpose(self.H) @ self.H).toarray()) )
        sr_im = self.EntireSR(in_im,H,HT,HTH)
        Py = self.sess.run(sr_im,feed_dict = {in_im:Py.reshape(Py.shape+(1,))})
        Py = np.squeeze(Py)
        res = RestoreImage( np.transpose(Py.reshape((Py.shape[0],-1))) ,self.scale*self.blk_size//2,hr_h,hr_w)
        return res
    
    R = srch(lr_image[:,:,0]).reshape((hr_h,hr_w,1))
    G = srch(lr_image[:,:,1]).reshape((hr_h,hr_w,1))
    B = srch(lr_image[:,:,2]).reshape((hr_h,hr_w,1))
    result = np.concatenate([R,G,B],axis = 2)
    return OverflowPro(result)

