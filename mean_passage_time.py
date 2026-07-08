# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:01:05 2026

@author: hibado
"""
import numpy as np
import matplotlib.pyplot as plt 
from itertools import combinations
import math 

num_robot=500
num_trial = 1000

S2 = np.random.rand()/10
# S2 = 1
M = np.zeros((num_robot,num_robot))
for i in range(num_robot):
    for j in range(num_robot):
        #j: informed 
        I = j+1
        n = num_robot - I # number of uninformed robot
        r = i - j  #increment of informed robot 
        if i>=j:
            M[i,j] = math.comb(n,r)*(1-(1-S2)**I)**r*((1-S2)**I)**(n-r)
Q = M[:num_robot-1, :num_robot-1]
v = np.zeros(num_robot-1)
v[0]=1
k_pred = np.ones(num_robot-1)@np.linalg.inv(np.eye(num_robot-1)-Q)@v

K =int(k_pred*2)

kc = np.ones(num_trial)*np.nan
for trial in range(num_trial):
    print(str(trial)+"/"+str(num_trial))
    informed = np.zeros(num_robot, dtype=bool)
    informed[0]=True
    for k in range(K):
        informed_new = informed.copy()
        for i,j in combinations(range(num_robot), 2): 
            if np.random.rand()<S2:
                state = informed[i] | informed[j]
                informed_new[i] = informed_new[i] | state.copy()
                informed_new[j] = informed_new[j] | state.copy()
        informed = informed_new.copy()        
        if (np.prod(informed)):
            kc[trial]= k
            break
    

#average percolation probability
k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    

p_preds=[]
for T in np.arange(1,K+1,1):
    p = sum(np.linalg.matrix_power(Q, T)@v)
    p_preds.append(1-p.copy())    


plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red" )
plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue")
plt.xlabel("K (time step)")
plt.ylabel("percolation probability")
# plt.vlines(np.log(num_robot)/((num_robot)*S2),0,1)
plt.vlines(k_pred,0,1, label = "Absorption Time ")
plt.vlines(np.mean(kc)+1,0,1, color="red", linestyle="--", label = "Mean K")

plt.title(f"single source percolation ({num_robot} agents)")