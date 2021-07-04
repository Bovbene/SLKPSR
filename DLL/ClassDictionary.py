# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:23:01 2020

@author: 月光下的云海
"""

import numpy as np
import pandas as pd

#字典类
"""===============================================================================================
Introduction: The PCA Dictionary Class (This class contain the method that convert dictionary into 
                                        dataframe)
---------------------------------------------------------------------------------------------------
Name: Dictionary
Input: dic = None,idx = None,centroid = None
		dic			    ----(numpy) dictionary array
		idx   		    ----(int) the index of current dictionary
        centroid        ----(numpy) the centroid of current class
---------------------------------------------------------------------------------------------------
Created on Fri Jun 19 14:23:01 2020
@author: 月光下的云海
"==============================================================================================="""
class Dictionary():
    def __init__(self,dic = None,idx = None,centroid = None):
        self.dic = dic
        self.idx = idx
        self.centroid = centroid
        
    def convert_to_dataframe(self):
        size = self.centroid.shape[0]
        idx = pd.DataFrame({
                  'idx'+str(self.idx):{0:self.idx}
                  })

        t_s = self.centroid
        t_s = t_s.reshape(size,)
        t_s = np.array(t_s)
        t_s = t_s.reshape(size,)
        centroid = pd.DataFrame({
                                  'centroid'+str(self.idx):t_s
                                  })
        dic = pd.DataFrame(self.dic)
        df = pd.concat([idx,centroid,dic],axis=1)
        return(df)

