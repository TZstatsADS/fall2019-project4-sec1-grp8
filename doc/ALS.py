#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:51:17 2019

@author: HenryZhou
"""

import numpy as np
import pandas as pd
import time
import math

rating = pd.read_csv('../data/ml-latest-small/ratings.csv')
len(rating['userId'].unique())
len(rating['movieId'].unique())

R = pd.read_csv('../data/ml-latest-small/matrix.csv')
R = R.to_numpy()
R = R[:,1:]

def nan_to_zero(R):
    X = np.where(~np.isnan(R), R, 0)
    return X
    
def zero_to_nan(R):
    X = R.copy()
    X[X == 0] = np.nan
    return X

def train_test_split(R, train_size = 0.8):
    """
    Args:
        R: Whole training data (np.ndarray)
        train_size: Proportion of training data
    
    Output:
        (TRAIN, TEST)
    """
        
    start_time = time.time()
        
    movies = dict()
    users = dict()
    
    N = np.count_nonzero(~np.isnan(R)) # number of non-Nan elements
    
    n = R.shape[0]
    m = R.shape[1]
        
    Rcopy = nan_to_zero(R)
    R0 = nan_to_zero(R)
    R1 = np.zeros((n,m))
    
    for i in range(n):
        Ii = [j for j,val in enumerate(R[i,:]) if not np.isnan(val)]
        train_set = np.random.choice(Ii, size = round(train_size*len(Ii)),
                                     replace = False)
        users[i] = len(train_set)
        
        for j in train_set:
            movies[j] = movies.get(j, 0) + 1
            R1[i,j] = R0[i,j]
            #train_row_index.append(i)
            #train_col_index.append(j)
    
    all_movies = set(np.arange(m))
    picked_movies = set(movies.keys())        
    
    missing_movies = all_movies.difference(picked_movies)
    
    if len(missing_movies) == 0: # all movies have been covered
        #TRAIN = Rcopy[(np.array(train_row_index), np.array(train_col_index))].reshape(n,m)
        #TEST = Rcopy - TRAIN
        #TRAIN = zero_to_nan(TRAIN)
        #TEST = zero_to_nan(TEST)
        R2 = R0 - R1
        R1 = zero_to_nan(R1)
        R2 = zero_to_nan(R2)
        NN = np.count_nonzero(~np.isnan(R1))
        prop = NN/N
        print('###### Size of training set: %f' % prop, ' ######')
        print('###### Running time: %s s ######' % (time.time()-start_time))       
        return (R1,R2)
    
    else:
        for j in missing_movies:
            Ij = np.nonzero(Rcopy[:,j])[0]
            add_user = np.random.choice(Ij, size = 1) 
            # new item: (add_user,j)
            drop_movie = max(movies, key = movies.get) 
            # drop one entry with most frequently rated movies
            drop_candidates = np.nonzero(R1[:,drop_movie])[0]
            if len(drop_candidates) == 0:
                print('Drop_movie: ', drop_movie)
                raise Exception('WRONG')
            user_subset = {k:users[k] for k in drop_candidates}
            drop_user = max(user_subset, key = user_subset.get)

            movies[drop_movie] -= 1
            movies[j] = 1
            users[drop_user] -= 1
            users[int(add_user)] += 1            
            R1[drop_user, drop_movie] = 0
            R1[add_user, j] = R0[add_user, j]
    
    R2 = R0 - R1
    R1 = zero_to_nan(R1)
    R2 = zero_to_nan(R2)
    NN = np.count_nonzero(~np.isnan(R1))
    prop = NN/N
    print('###### Size of training set: %f' % prop, ' ######')
    print('###### Running time: %s s ######' % (time.time()-start_time)) 
    return (R1,R2)

def RMSE(X, Y):
    """
    Compute the RMSE of between X and Y
    """
    if X.shape != Y.shape:
        raise Exception ('X and Y must be in the same shape!')
        
    temp = X.flatten() - Y.flatten()
    rmse = math.sqrt(sum(temp**2))
    return rmse
    
    
def ALS(R, d, Lambda = 0.05, stop_criterion = 0.01, max_iter = 1000):
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
    
    n = R.shape[0]
    m = R.shape[1]
    
    U = np.random.uniform(0, 0.5, (n,d)) # users
    M = np.random.uniform(0, 0.5, (d,m)) # movies
    
    ## Initialize M
    M[0,:] = np.nanmean(R, axis=0)
    
    bu = np.nanstd(R, axis=1) # bias for users
    bm = np.nanstd(R, axis=0) # bias for movies
    mu = np.nanmean(R)
    
    #R0 = nan_to_zero(R)
    T = R.copy()
    T = nan_to_zero(T)
    temp = np.nonzero(T)
    
    for (i,j) in zip(list(temp[0]), list(temp[1])):
        T[i,j] = R[i,j] - mu - bu[i] - bm[j] # new rating matrix combining bias
        
    error = 1
    diff = 1000 * stop_criterion
    count = 1
    
    while diff > stop_criterion and count < (max_iter+1):
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
        print('Updating M complete')
        
        diff = abs(RMSE(T, np.matmul(U,M)) - error)
        error = RMSE(T, np.matmul(U,M))
        print('RMSE = %f' % error)
        count += 1
    
    return (U,M)
        
U,M = ALS(R, d = 5)
