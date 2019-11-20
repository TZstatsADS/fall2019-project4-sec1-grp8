#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:47:44 2019

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

def train_test_split(R, train_size = 0.8, seed = None):
    """
    Args:
        R: Whole training data (np.ndarray)
        train_size: Proportion of training data
    
    Output:
        (TRAIN, TEST)
    """
    
    if seed:
        np.random.seed(seed)
        
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
    return (R1,R2)