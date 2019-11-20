#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:24:42 2019

@author: Justine Zhang
"""
import numpy as np

def nan_to_zero(R):
    X = np.where(~np.isnan(R), R, 0)
    return X

def cosine_similarity(v1, v2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5)*(normB**0.5))

def similarity(i,train,test,M):
    """
    Generate the similarity dictionary of each testing movie for user i
    
    Args:
        train: training set (np.ndarray)
        test: testing set (np.ndarray)
        M: the movie matrix generated from ALS.py
    
    Output:
        simi (array of dictionary, length: number of rated testing movies for user i)
        
    Note:
        For each dictionary sim in simi, the value is the index of a training movie
        and the key is the cosine similarity between this training movie and the movie we want to test
    """
    simi = []
    train_idx = np.nonzero(train[i,:])[0]
    test_idx = np.nonzero(test[i,:])[0]
    for te in test_idx:
        sim = dict()
        v1 = M[:,te]
        for tr in train_idx:
            v2 = M[:,tr]
            s = cosine_similarity(v1,v2)
            sim[tr] = s
        simi.append(sim)
    return simi


def knn(K, train, test, M, ss):
    """    
    Args:
        K: number of top neighbors
    
    Output:
        pred: the prediction score for each testing movie
        
    Note:
        For each user, if the number of rated training movies is less than K, we calculate the mean of 
        all the training movies' ratings as the prediction score for testing movies.
    """    
    #print("######### Start Finding the Optimal K #########")
    n,m = test.shape
    pred = np.zeros((n,m))
    for i in range(n):
        simi = ss[i]
        train_idx = np.nonzero(train[i,:])[0]
        test_idx = np.nonzero(test[i,:])[0]  
        len_train = len(train_idx)
        if len_train <= K:
            for j in range(0, len(simi)):
                sim = simi[j]
                idx = test_idx[j]
                w = sorted(sim.keys(), reverse=True)
                v = sorted(sim, key = lambda i: sim[i], reverse=True)
                pred[i,idx] = np.average(train[i,v], weights=w)
                                               
        else:
            for j in range(0,len(simi)):
                sim = simi[j]
                idx = test_idx[j]
                w = sorted(sim.values(), reverse=True)[:K]
                v = sorted(sim, key = lambda i: sim[i], reverse=True)[:K]
                pred[i,idx] = np.average(train[i,v], weights=w)
                                           
    return pred