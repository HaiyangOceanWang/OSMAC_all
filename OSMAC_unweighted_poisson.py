# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Algorithm 3
More efficient estimation based on Poisson sampling
"""

import numpy as np
import tools


def algorithmThreeMain(X, y, r):
    pSSP, c0, c1 = tools.pilotSubSampProb(y)
    
    X_poi_ssp, y_poi_ssp, probs_poi_ssp = tools.poissonSample(X, y, pSSP, 200)
    
    beta1 = tools.poissonUnwNewton(X_poi_ssp, y_poi_ssp, probs_poi_ssp)
    beta1_hat = beta1.copy()
    beta1_hat[0] = beta1_hat[0] + np.log(c0/c1)
    
    Npsi = y.shape[0] * tools.psi(X_poi_ssp, y_poi_ssp, probs_poi_ssp, beta1_hat)

    poi_probs = tools.poissonSampleProbs(X, y, beta1_hat, Npsi)    
    
    
    X_poi, y_poi, probs_poi = tools.poissonSample(X, y, poi_probs, r)
    
    beta_p = tools.poissonUnwNewton(X_poi, y_poi, probs_poi)
    beta_p_hat = beta_p.copy()
    beta_p_hat = beta_p_hat + beta1_hat
    
    beta = tools.combineTwoBeta(X_poi_ssp, beta1, beta1_hat, X_poi, beta_p, beta_p_hat)
    
    return beta

