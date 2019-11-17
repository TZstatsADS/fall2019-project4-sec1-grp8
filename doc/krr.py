#!/usr/bin/env python
# coding: utf-8

"""
Created on Sat Nov 16 23:46:50 2019
@author: Suzy Gao
"""


import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def nan_to_zero(R):
    X = np.where(~np.isnan(R), R, 0)
    return X

# Use train, test from ALS
train = np.load("../data/train.npy")
test = np.load("../data/test.npy")
train = nan_to_zero(train)
test = nan_to_zero(test)
M = np.load("../data/M.npy")
print("M.shape:",M.shape)


n = train.shape[0]
m = train.shape[1]
M1 = pd.DataFrame(M)
print(n)
print(m)


#define X_train, y_train, X_test, y_test to fit the krr model and then predict
def train_test_build(train,test,i,M1):
    X_train = pd.DataFrame([])
    X_test = pd.DataFrame([])
    y_train = []
    y_test = []
    train_idx = np.nonzero(train[i,:])[0]
    test_idx = np.nonzero(test[i,:])[0]
    for tr in train_idx:
        X_train = X_train.append(M1[tr])
        y_train.append(train[i,tr])
        for te in test_idx:
            X_test = X_test.append(M1[te])
            y_test.append(test[i,te])
    return X_train, y_train, X_test, y_test


data = []
for i in range(n):
    print(i,"-",n)
    df = train_test_build(train,test,i,M1)
    data.append(df)



#kernel ridge regression with rbf kernel
def krr_rbf(a,d,g,c):
    acc_score = []
    for i in range(n):   
        krr = KernelRidge(kernel='rbf', alpha=a, degree=d, gamma=g, coef0=c)
        krr.fit(data[i][0], data[i][1])
        y_krr = krr.predict(data[i][2])   
        acc_score.append(np.sqrt(mean_squared_error(data[i][3], y_krr))) 

    avg_acc_score = sum(acc_score)/len(acc_score)
    print('Average RMSE for krr_rbf is:', avg_acc_score)



#kernel ridge regression with linear kernel
def krr_linear(a,d,c):
    acc_score = []
    for i in range(n):   
        krr = KernelRidge(kernel='linear', alpha=a, degree=d, gamma=None, coef0=c)
        krr.fit(data[i][0], data[i][1])
        y_krr = krr.predict(data[i][2])   
        acc_score.append(np.sqrt(mean_squared_error(data[i][3], y_krr))) 

    avg_acc_score = sum(acc_score)/len(acc_score)
    print('Average RMSE for krr_linear is:', avg_acc_score)



#kernel ridge regression with poly kernel
def krr_poly(a,d,g,c):
    acc_score = []
    for i in range(n):   
        krr = KernelRidge(kernel='poly', alpha=a, degree=d, gamma=g, coef0=c)
        krr.fit(data[i][0], data[i][1])
        y_krr = krr.predict(data[i][2])   
        acc_score.append(np.sqrt(mean_squared_error(data[i][3], y_krr))) 

    avg_acc_score = sum(acc_score)/len(acc_score)
    print('Average RMSE for krr_poly is:', avg_acc_score)






