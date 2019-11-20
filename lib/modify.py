#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:34:35 2019

@author: HenryZhou
"""
import numpy as np

def nan_to_zero(R):    
    X = np.where(~np.isnan(R), R, 0)
    return X
    
def zero_to_nan(R):
    X = R.copy()
    X[X == 0] = np.nan
    return X

def modify(R):
    mu = np.nanmean(R)
    bu = np.nanmean(R, axis=1) - mu # bias for users
    bm = np.nanmean(R, axis=0) - mu # bias for movies
    
    R0 = nan_to_zero(R)
    index = np.nonzero(R0)
    for i,j in zip(index[0],index[1]):
        R0[i,j] -= mu+bu[i]+bm[j]
    R0 = zero_to_nan(R0)
    return R0