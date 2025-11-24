# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 11:43:06 2025

@author: hibado
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt 

data = {}
for i in range(3):
    with open("data/"+str(i)+'.p', 'rb') as file:
       data[i] = pickle.load(file)
       
x =  np.linspace(2,250, 1000)
# plt.plot(x,np.exp(2.55)*(np.log(x)**2/(x)**0.81)*(-np.log(pTp))**0.5, '--', color="r")
# plt.plot(x,np.exp(3.35)*(np.log(0.2*x+0.8)**2/(x)**0.83)*(-np.log(pTp))**1, '--', color="r")
for dat in data.values():
    p = dat["p"]
    K = dat["K"]
    num_robots = dat["N"]
    pTp = np.inner(p,p)
    plt.figure()
    plt.plot(x,5.5*(np.log(0.3*x+0.7)**2/(x)**((1-pTp)))*(-np.log(pTp)), '--', color="r")
    
    plt.plot(2,1/pTp-1, 'x', color="r")
    
    plt.errorbar(num_robots, np.mean(K, axis=1), np.std(K, axis=1)/np.sqrt(30), fmt='.')
    plt.plot(num_robots, np.quantile(K, 0.75, axis=1), '.', color="b", alpha=0.1)
    plt.plot(num_robots, np.quantile(K, 0.25, axis=1), '.', color="b", alpha=0.1)
    plt.xlim([0,250])
    plt.xlabel(r"$N$")
    plt.ylabel(r"$K_c$")
    # plt.yticks(np.arange(0,int(np.nanmax(np.quantile(critical, 0.75, axis=1))),3))
    plt.title("Kc vs number of robots, pTp = " + str(pTp))
    plt.legend()