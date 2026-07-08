# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 15:51:14 2025

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt 
from itertools import combinations

num_regions=50
num_robot = 500
num_trial=1000
K = 20
states=[]
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
p = np.ones(num_regions)
p = p/sum(p)
S2 = np.sum(p**2)
S3  = np.sum(p**3)
a = 1-2*S2+S3


kc = np.ones(num_trial)*np.nan
informed_percentage = np.ones((num_trial, K))

for trial in range(num_trial):
    print(str(trial)+"/"+str(num_trial))
    state = np.eye(num_robot, dtype=bool)
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                state[i,:] = state[i,:] | state[j,:]
                state[j,:] = state[i,:] | state[j,:]
        informed_percentage[trial,k] = np.sum(state[:,0])/num_robot
        # informed_percentage[trial,k] = np.min(np.sum(state, axis=1)/num_robot)
        if (np.prod(state[:,0])):
            kc[trial]= k
            break
    

k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    


plt.figure(dpi=800)
plt.plot(np.arange(1,K+1,1), np.mean(informed_percentage, axis=0), color = "red", label="simulations" )
plt.xlabel("K (time step)")
plt.ylabel("informed percentage")
plt.title("single source percolation ("+ str(num_robot)+"agents)")

x0 = 1/num_robot
x=[x0]
for k in range(20):
    dx=0
    xk = x[-1]
    for pi in p:
        dx+= (1-(1-pi)**(xk*num_robot))*pi
    dx*=(1-xk)
        
    x.append((xk+dx))

plt.plot(x,"--", color = "blue", label="Mean-field-approx.")
plt.legend()