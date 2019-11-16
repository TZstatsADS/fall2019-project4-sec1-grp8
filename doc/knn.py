#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:24:42 2019

@author: Justine Zhang
"""
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

rating = pd.read_csv('../data/ml-latest-small/ratings.csv')
print("userId:",len(rating['userId'].unique()))
print("movieId:",len(rating['movieId'].unique()))

def nan_to_zero(R):
    X = np.where(~np.isnan(R), R, 0)
    return X

# We save train, test, and M in ALS.py
train = np.load("../data/train.npy")
test = np.load("../data/test.npy")
train = nan_to_zero(train)
test = nan_to_zero(test)
M = np.load("../data/M.npy")
print("M.shape:",M.shape)

n = train.shape[0]
m = train.shape[1]

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
            sim[s] = tr
        simi.append(sim)
    return simi

# store all the similarity calculations in an array ss to reduce complexity
print("######### Start Calculating the similarity between movies #########")
ss = []
for i in range(n):
    print(i,"/",n)
    simi = similarity(i,train,test,M)
    ss.append(simi)

def knn(K,test,M,n,m,ss):
    """    
    Args:
        K: number of top neighbors
    
    Output:
        pred: the prediction score for each testing movie
        
    Note:
        For each user, if the number of rated training movies is less than K, we calculate the mean of 
        all the training movies' ratings as the prediction score for testing movies.
    """    
    print("######### Start Finding the Optimal K #########")
    pred = np.zeros((n,m))
    for i in range(n):
        simi = ss[i]
        train_idx = np.nonzero(train[i,:])[0]
        test_idx = np.nonzero(test[i,:])[0]  
        len_train = len(train_idx)
        if len_train <= K:
            avg = np.mean(train[i,train_idx])
            pred[i,test_idx] = avg   
        else:
            for j in range(0,len(simi)):
                sim = simi[j]
                idx = test_idx[j]
                a = sorted(sim.keys(), reverse=True)
                a = a[:k]
                v = [sim[j] for j in a]
                pred[i,idx] = np.mean(train[i,v])  
    return pred

# Find the best k
Ks = np.arange(5, 20, 1) 
rmse = np.zeros(len(Ks)) 

for i, k in enumerate(Ks): 
    print('K:',k)
    pred = knn(k,test,M,n,m,ss)
    temp = pred.flatten() - test.flatten()
    error = math.sqrt(sum(temp**2))
    print('RMSE = %f' % error)
    rmse[i] = error

#Generate plot 
plt.plot(Ks, rmse, label = 'Testing Dataset RMSE') 

plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('RMSE') 
plt.show()
