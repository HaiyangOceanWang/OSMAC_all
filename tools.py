# -*- coding: utf-8 -*-
"""
tools for all OSMAC-like library
"""

import numpy as np


def logistic_func(beta, X):
    return 1 / (1 + np.exp(-np.dot(X,beta)))

def subsample(X, y,r, probs):
    subsampledRows = np.random.choice(X.shape[0], r, p=probs)
    X_ssp = X[subsampledRows]
    y_ssp = y[subsampledRows]
    probs_ssp = probs[subsampledRows]
    return X_ssp, y_ssp, probs_ssp

def unweightedNewton(X, y, converge_rate=.000001):
    beta0 = np.repeat(0, X.shape[1])
    dist = 1
    while(dist > converge_rate):
        p = logistic_func(beta0, X)
        H = (X.T * (p * (1 - p))).dot(X)
        J = np.sum(X.T * ((p - y)), axis=1)
        try:
            HJ = np.linalg.solve(H, J)
            beta1 = beta0 - HJ
        except:
            #H_inv = np.linalg.pinv(H);beta1 = beta0 - H_inv.dot(J)
            beta1 = beta0 - np.linalg.lstsq(H, J)[0]
            dist = np.linalg.norm(beta1 - beta0)
            beta0 = beta1
        else:
            dist = np.linalg.norm(beta1 - beta0)
            beta0 = beta1
    return beta1

def pilotSubSampProb(y):
    priorP = np.count_nonzero(y) / y.shape[0]#Approx Prior MargProb of y from training set
    c0 = 1 / (2 * (1 - priorP))
    c1 = 1 / (2 * priorP) 
    
    pilotSubSampProb = (c0 * (1 - y) + c1 * y) / y.shape[0]
    return pilotSubSampProb, c0, c1

def osmacProb(X, y, beta, h_func='mVc'):
    p = logistic_func(beta, X)
    dif = np.abs(y - p) 
    Xnorm = np.linalg.norm(X, axis=1)
    pis = dif * Xnorm
    osmacProb = pis / sum(pis)
    return osmacProb

def l(x,beta):
    p = logistic_func(beta,x)
    H = (x.T*(p*(1-p))).dot(x)
    return np.sum(H)

def combineTwoBeta(x1, beta1, beta1_hat, x, beta_r, beta_r_hat):
    l1 = l(x1, beta1)
    lr = l(x, beta_r)
    beta_final = (l1 * beta1_hat + lr * beta_r_hat ) / (l1 + lr)
    return beta_final

