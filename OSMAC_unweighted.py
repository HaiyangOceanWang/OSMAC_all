# -*- coding: utf-8 -*-
"""
Algorthm 2
More efficent  estimation based on subsampling with replacement

"""
import numpy as np
import tools

def algorithmTwoMain(X, y, r):
    
    pSSP, c0, c1 = tools.pilotSubSampProb(y)
    
    X_ssp, y_ssp, _ = tools.subsample(X, y, 200, pSSP)
    beta1 = tools.unweightedNewton(X_ssp, y_ssp)
    beta1_hat = beta1.copy()
    beta1_hat[0] = beta1_hat[0] + np.log(c0 / c1)
    
    
    osmacSSP = tools.osmacProb(X, y, beta1)
    X_osmac, y_osmac, _= tools.subsample(X, y, r, osmacSSP)
        
    beta_r = tools.unweightedNewton(X_osmac, y_osmac)
    beta_r_hat = beta_r.copy()
    beta_r_hat = beta_r_hat + beta1_hat
    
    beta = tools.combineTwoBeta(X_ssp, beta1, beta1_hat, X_osmac, beta_r, beta_r_hat)
    
    return beta
