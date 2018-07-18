# -*- coding: utf-8 -*-
"""
Make simulated datasets with true beta = np.repeat(0.5, d)
"""
import numpy as np
import random
from tools import logistic_func

def makeNormalData(d = 7, n = 10000, mean = 0):
    random.seed(1)
    beta = np.repeat(0.5, d)
    
    sig = np.zeros((d, d))
    for row in range(d):
        for col in range(d):
            if row == col:
                sig[row][col] = 1
            else: 
                sig[row][col] = 0.5      
        
    X = np.random.multivariate_normal(np.repeat(mean,d), sig, size=n)
    p = logistic_func(beta, X)
    y = np.random.binomial(1, p)
    
    return X, y, p, beta


