#!/usr/bin/env python
# coding: utf-8

"""
Created on Sat Nov 16 23:46:50 2019
@author: Suzy Gao
"""


import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def nan_to_zero(R):
    X = np.where(~np.isnan(R), R, 0)
    return X

def zero_to_nan(R):
    X = R.copy()
    X[X == 0] = np.nan
    return X

#kernel ridge regression with rbf kernel
def krr_rbf(i, train, test, M, alpha, gamma, limit=None):
    if limit:
        (Xtrain,Ytrain),(Xtest,Ytest) = train_test_build(i,train,test,M,limit)
    else:
        (Xtrain,Ytrain),(Xtest,Ytest) = train_test_build(i,train,test,M)
    
    clf = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    clf.fit(Xtrain,Ytrain)
    pred_test = clf.predict(Xtest)
    pred_train = clf.predict(Xtrain)
    train_error = sum((pred_train-Ytrain)**2)
    test_error = sum((pred_test-Ytest)**2)
    
    return (train_error, test_error)

#kernel ridge regression with linear kernel
def krr_linear(i, train, test, M, alpha, limit=None):
    if limit:
        (Xtrain,Ytrain),(Xtest,Ytest) = train_test_build(i,train,test,M,limit)
    else:
        (Xtrain,Ytrain),(Xtest,Ytest) = train_test_build(i,train,test,M)
    
    clf = KernelRidge(alpha=alpha)
    clf.fit(Xtrain,Ytrain)
    pred_test = clf.predict(Xtest)
    pred_train = clf.predict(Xtrain)
    train_error = sum((pred_train-Ytrain)**2)
    test_error = sum((pred_test-Ytest)**2)
    
    return (train_error, test_error)

def train_test_build(i, train, test, M, limit = None):
    """
    Constructs training and testing set for KRR
    
    Args:
        i: user to be predicted
        train: Original training set
        test: Original testing set
        M: movie feature matrix
        
        if impose_limit == True: only use limit most frequently rated movies
    
    Return:
        ((Xtrain,Ytrain), (Xtest,Ytest))
    """
    
    def normalize(x):
        norm = np.linalg.norm(x)
        return [i/norm for i in x]
    
    if np.isnan(train).any():
        train = nan_to_zero(train)
    if np.isnan(test).any():
        test = nan_to_zero(test)
        
    train_index = np.nonzero(train[i,:])[0]
    test_index = np.nonzero(test[i,:])[0]
    
    if limit:
        temp = zero_to_nan(train)        
        n_rating = {m:sum(~np.isnan(temp[:,m])) for m in train_index}
        train_index = sorted(n_rating, key = lambda i: n_rating[i], reverse=True)[:limit]
    
    Xtrain = np.transpose(M[:, train_index])
    Xtrain = np.apply_along_axis(normalize, 1, Xtrain)
    Xtest = np.transpose(M[:, test_index])
    Xtest = np.apply_along_axis(normalize, 1, Xtest)
    Ytrain = train[i, train_index]
    Ytest = test[i, test_index]
    
    return ((Xtrain, Ytrain), (Xtest, Ytest))