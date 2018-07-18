# -*- coding: utf-8 -*-
"""
Make MSE plots

"""
import time
import matplotlib.pyplot as plt
import OSMAC_unweighted,OSMAC_unweighted_poisson
import makeSimulatedData

X, y, _, beta = makeSimulatedData.makeNormalData(d = 7, n = 10000, mean = 0)

start = time.clock()

s = 1000
r = [100, 200, 300, 500, 700, 1000]

beta_mVc, beta_p_mVc = [], []
mse_mVc, mse_beta_p_mVc = [], []
for times in r:   
    for j in range(s):      
        beta_mVc.append(OSMAC_unweighted.algorithmTwoMain(X, y, times))
        beta_p_mVc.append(OSMAC_unweighted_poisson.algorithmThreeMain(X, y, times))

    mse_mVc.append(((beta_mVc-beta) ** 2).sum(axis = 1).mean())
    mse_beta_p_mVc.append(((beta_p_mVc-beta) ** 2).sum(axis = 1).mean())

    beta_mVc, beta_p_mVc = [], []          

##plot
fig = plt.figure()
#plt.plot(r,mse_uni,'kh-', label="uniform")
#plt.plot(r,mse_mMSE,'rx-', label="mMSE")
plt.plot(r, mse_mVc, 'bo--', label="MSE-mVc-beta-r")
plt.plot(r, mse_beta_p_mVc, 'rx--', label="MSE-mVc-beta-p")
#plt.plot(r,mse_LCC,'g2-', label="LCC")
#plt.plot(r,mse_full,'y3-',color='gray', label="full")
plt.xlabel("r")
plt.ylabel("MSEs")
plt.title(" OSMAC_unweighted")
plt.legend()


'''Time used'''
elapsed = (time.clock() - start)
print("Time used:",time.strftime("%H:%M:%S", time.gmtime(elapsed)),"seconds")
