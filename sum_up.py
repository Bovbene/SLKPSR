# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:19:15 2021

@author: 月光下的云海
"""

from glob import glob
from PIL import Image
import pandas as pd
import numpy as np
from DLL.Valuation import PSNR as psnr
from DLL.Valuation import ssim
import os
import cv2 as cv

def get_name(p):
    fn = os.path.split(p)[-1]
    fn = os.path.splitext(fn)[0]
    return fn

def get_sr_name(p):
    fn = get_name(p)
    loc = fn.index('_')
    return fn[:loc]

def sum_up_on_model(model):
    res_path = "./RESULT/"+model+"/"
    
    fn_li = os.listdir(res_path)
    df_li = []
    for data_set in fn_li:
        
        sr_img_path = res_path+data_set+'/*.png'
        hr_img_path = "./DATABASE/"+data_set+"/*.bmp"
        
        sr_path = glob(sr_img_path)
        hr_path = glob(hr_img_path)
        
        if len(sr_path) == 0 or len(hr_path) == 0:
            raise ValueError("Dataset Error!")
        
        df = pd.DataFrame([],columns = ['name','psnr','ssim'])
        
        for p1,p2 in zip(sr_path,hr_path):
            
            fn = get_name(p1)
            fn = fn[:fn.index('_')]
            
            # if fn != get_name(p2):
            #     raise AttributeError("Match Error!")
  			
            nline = {'name':fn}
            
            sr = np.array(Image.open(p1))
            hr = np.array(Image.open(p2))
  			
            if sr.shape != hr.shape:
                sr = cv.resize(sr,(hr.shape[1],hr.shape[0]),interpolation = cv.INTER_CUBIC)
            
            ipsnr,issim = psnr(sr,hr,4),ssim(hr,sr)
  			
            
            nline['psnr'] = ipsnr
            nline['ssim'] = issim
  			
            df = df.append(nline,ignore_index = True)
            
        df_li.append(df)
        
    with pd.ExcelWriter('./RESULT/'+model+'_res_statistic.xlsx') as writer:
        for name,df in zip(fn_li,df_li):
            df.to_excel(writer,index = False,sheet_name = name)

if __name__ == '__main__':
    
    all_model = ["VDSRx1"]
    for model in all_model:
        sum_up_on_model(model)
    
        



