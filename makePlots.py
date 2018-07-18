# -*- coding: utf-8 -*-
"""
Make MSE or var-cor plots 

"""
import time
import matplotlib.pyplot as plt
import OSMAC_unweighted
import makeSimulatedData

X, y, p, beta = makeSimulatedData.makeNormalData(d = 7, n = 10000, mean = 0)

start = time.clock()

s = 1000
r = [100, 200, 300, 500, 700, 1000]
#beta_f = unweightedNewton(X, y)
beta_uni, beta_mVc, beta_mMSE, beta_full = [], [], [], []
mse_uni, mse_mVc, mse_mMSE, mse_full = [], [], [], []
for times in r:   
    for j in range(s):      
        beta_mVc.append(OSMAC_unweighted.algorithmTwoMain(X, y, times))
        #beta_mMSE.append(om.two_steps(x,y,r0,times,'mmse'))
        #beta_uni.append(om.two_steps(x,y,r0,times,'uni'))
        #beta_full.append(beta_f)
    mse_mVc.append(((beta_mVc-beta) ** 2).sum(axis = 1).mean())
    #mse_mMSE.append(((beta_mMSE-beta)**2).sum(axis=1).mean())
    #mse_uni.append(((beta_uni-beta)**2).sum(axis=1).mean())
    #mse_full.append(((beta_full-beta)**2).sum(axis=1).mean())
    beta_uni, beta_mVc, beta_mMSE, beta_full=[], [], [], []          

##plot
fig = plt.figure()
#plt.plot(r,mse_uni,'kh-', label="uniform")
#plt.plot(r,mse_mMSE,'rx-', label="mMSE")
plt.plot(r, mse_mVc, 'bo--', label="mVc")
#plt.plot(r,mse_LCC,'g2-', label="LCC")
#plt.plot(r,mse_full,'y3-',color='gray', label="full")
plt.xlabel("r")
plt.ylabel("MSE")
plt.title(" OSMAC_unweighted")
plt.legend()


'''Time used'''
elapsed = (time.clock() - start)
print("Time used:",time.strftime("%H:%M:%S", time.gmtime(elapsed)),"seconds")
