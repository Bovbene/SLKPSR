# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:45:53 2020

@author: 月光下的云海
"""

import numpy as np

PI = np.pi

'''============================================================================
FUNCTION:overDct
FEATURE: 构建一个过完备的DCT字典
INPUTS:
   raw-------字典的行数
   column----字典的列数
OUTPUTS:
   过完备DCT字典A
Tip：raw,column必须开方为整数
============================================================================'''  
def overDct(raw,column):
    MM = raw**0.5;NN = column**0.5;
    A1 = np.matrix([i for i in range(0,int(MM))])
    A2 = np.matrix([i for i in range(0,int(NN))])
    A = ((2/MM)**0.5)*np.cos((PI/(2*MM))*(np.transpose(A1)*A2))
    A[0,:] = A[0,:]/(2**0.5)
    A = np.kron(A,A)
    return(A)


