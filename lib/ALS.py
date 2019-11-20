#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:38:26 2019

@author: HenryZhou
"""
import numpy as np
import time
import math

def nan_to_zero(R):    
    X = np.where(~np.isnan(R), R, 0)
    return X
    
def zero_to_nan(R):
    X = R.copy()
    X[X == 0] = np.nan
    return X

def RMSE(R, Rhat):
    """
    Compute the RMSE of between R and Rhat
    """
    
    if np.isnan(R).any():
        R = nan_to_zero(R)
        
    if R.shape != Rhat.shape:
        raise Exception ('X and Y must be in the same shape!')
        
    index = np.nonzero(R)
    n = len(index[0])
    rmse = 0
    for i,j in zip(list(index[0]), list(index[1])):
        rmse += (R[i,j]-Rhat[i,j])**2
    
    return math.sqrt(rmse/n)
    
    
def ALS(R, d, Lambda = 0.05, stop_criterion = 0.0001, max_iter = 1000, seed = None,
        print_result = False):
    """
    Alternating Least Squares
    
    Args:
        R: Input matrix (n by m) to be factorized (np.ndarray)
        d: Dimension of feature space
        Lambda: Coefficient of regularization. Default to 0.05
        stop_deriv: stopping criterion based on RMSE
    
    Output:
        (U,M) U (n by d), M(d by m)
    """
    
    assert isinstance(d, int) # check if d is an integer
    
    if seed:
        np.random.seed(seed)
    
    n = R.shape[0]
    m = R.shape[1]
    
    U = np.random.uniform(0, 0.5, (n,d)) # users
    M = np.random.uniform(0, 0.5, (d,m)) # movies
    
    ## Initialize M
    M[0,:] = np.nanmean(R, axis=0)
    
    mu = np.nanmean(R)
    bu = np.nanmean(R, axis=1) - mu # bias for users
    bm = np.nanmean(R, axis=0) - mu # bias for movies
    
    #R0 = nan_to_zero(R)
    T = R.copy()
    T = nan_to_zero(T)
    temp = np.nonzero(T)
    
    for (i,j) in zip(list(temp[0]), list(temp[1])):
        T[i,j] = R[i,j] - mu - bu[i] - bm[j] # new rating matrix combining bias
        
    error = 100
    diff = 1000 * stop_criterion
    count = 1
    stepwise_error = [] # record stepwise error
    
    while diff > stop_criterion and count < (max_iter+1):
        if print_result:
            print('######## Iteration %d ########' % count)
              
        for i in range(n):
            # updating U
            user_i = T[i,:]
            Ii = np.nonzero(user_i)[0]
            n_ui = len(Ii)
            M_Ii = M[:,Ii]
            Ri_Ii = T[i,Ii]
            Ai = np.matmul(M_Ii, np.transpose(M_Ii)) + Lambda*n_ui*np.identity(d)
            Vi = np.matmul(M_Ii, np.transpose(Ri_Ii))
            U[i,:] = np.matmul(np.linalg.inv(Ai), Vi)
        if print_result:
            print('Updating U complete')
        
        for j in range(m):
            # updating V
            movie_j = T[:,j]
            Ij = np.nonzero(movie_j)[0]
            n_mj = len(Ij)
            U_Ij = U[Ij,:]
            R_Ij_j = T[Ij,j]
            Aj = np.matmul(np.transpose(U_Ij),U_Ij) + Lambda*n_mj*np.identity(d)
            Vj = np.matmul(np.transpose(U_Ij), R_Ij_j)
            M[:,j] = np.matmul(np.linalg.inv(Aj), Vj)
        if print_result:
            print('Updating M complete')
        
        Rhat = np.matmul(U,M)
        Rhat += mu        
        for i in range(n):
            Rhat[i,:] += bu[i]
        for j in range(m):
            Rhat[:,j] += bm[j]

        
        diff = abs(RMSE(R, Rhat) - error)
        error = RMSE(R, Rhat)
        stepwise_error.append(error)
        if print_result:
            print('RMSE = %f' % error)
        count += 1
    
    return ((U,M), stepwise_error)