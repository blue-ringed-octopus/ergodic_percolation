# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:33:27 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import math
from scipy.stats import dirichlet
from graph_planner import REMC, FMMC
import networkx as nx
from tqdm import tqdm

num_regions = 2
num_robot = 3
num_trial = 10000
K=5
rho0 = dirichlet.rvs(np.ones(num_regions)).T
P = dirichlet.rvs(np.ones(num_regions), 2).T

Q11 = np.array([[P[0,0]**2*P[1,1], P[0,1]**2*P[1,0]],
                [P[0,1]*P[1,0]**2, P[0,0]*P[1,1]**2]])

Q21 = np.array([[2*P[0,0]*P[1,0]*P[1,1], 2*P[1,0]*P[1,1]*P[0,1]],
                [2*P[0,1]*P[1,1]*P[1,0], 2*P[1,1]*P[0,1]*P[0,0]]])

Q22 = np.array([[P[0,0]*P[1,1]**2, P[0,1]*P[1,0]**2],
                [P[0,1]**2*P[1,0], P[0,0]**2*P[1,1]]])

P_combined = np.eye(5)
P_combined[0:2,0:2] = Q11
P_combined[2:4,2:4] = Q22
P_combined[2:4,0:2] = Q21
P_combined[-1,:] = 1- np.sum(P_combined[0:4,:], axis=0)
Q31 = P_combined[-1,0:2]
Q32 = P_combined[-1,2:4]

rho_combined = np.array([rho0[0]**2*rho0[1], rho0[0]*rho0[1]**2, 2*rho0[0]*rho0[1]**2, 2*rho0[0]**2*rho0[1], (rho0[0]**3+rho0[1]**3)])

I=[]
for k in range(K):
    I.append(np.linalg.matrix_power(P_combined, k)@rho_combined)
I = np.array(I)

percolation_percent_expected = []
for i in I:
    I1 = sum(i[0:2])
    I2 = sum(i[2:4])
    I3 = i[4]
    percolation_percent_expected.append((I1+2*I2+3*I3)/3)
    
lambda_,_ = np.linalg.eig(P)
lambda2 = abs(np.sort(lambda_)[0])
#%% lump approx 
d, v = np.linalg.eig(Q11)
d= np.abs(d)
v11 = v[:,d.argmax()]
v11 = v11/np.sum(v11)

d, v = np.linalg.eig(Q22)
d= np.abs(d)
v22 = v[:,d.argmax()]
v22 = v22/np.sum(v22)

Q11_lump = np.sum(Q11@v11)
Q21_lump = np.sum(Q21@v11)
Q22_lump = np.sum(Q22@v22)
Q31_lump =  np.sum(Q31@v11)
Q32_lump =  np.sum(Q32@v22)

Q_lump = np.eye(3)
Q_lump[0,0] = Q11_lump 
Q_lump[1,0] = Q21_lump 
Q_lump[1,1] = Q22_lump 
Q_lump[-1,:] = 1- np.sum(Q_lump[0:2,:], axis=0)
rho_lumped = np.zeros(3)
rho_lumped[0] = sum(rho_combined[0:2])
rho_lumped[1] = sum(rho_combined[2:4])
rho_lumped[2] = rho_combined[4]

I_lumped=[]
for k in range(K):
    I_lumped.append(np.linalg.matrix_power(Q_lump, k)@rho_lumped)
I_lumped = np.array(I_lumped)

percolation_percent_lumped = []
for i in I_lumped:
    percolation_percent_lumped.append((i[0]+2*i[1]+3*i[2])/3)
#%% simulation
kc_single = np.ones(num_trial)*np.nan
rho = np.zeros((num_trial, K, num_regions))
I_sim = np.zeros((num_trial,K, num_regions))
for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=rho0.T[0])
    for k in range(K): 
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                info = state[i,:].copy() |  state[j,:].copy() 
                state[i,:] = info
                state[j,:] = info
        if (np.prod(state[:,0])):
            kc_single[trial]= np.nanmin([k,kc_single[trial]])
            
        for robot in range(num_robot):
            I_sim[trial,k,r[robot]]+=state[robot,0]

            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]    
            # r[robot] = np.random.choice(range(num_regions), 1, P=pk[k+1])[0]

            rho[trial,k,r[robot]]+=1/num_robot
        # r[0] = np.random.choice(range(num_regions), 1, P=np.linalg.matrix_power(P, k+1)@I[0])

rho = np.mean(rho,axis=0)
k_arr_single = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc_single[i]):
        k[int(kc_single[i]):]=1
    k_arr_single.append(k)    
    
K_plot = K
plt.figure(dpi=800)
plt.plot(np.arange(0,K_plot,1), np.mean(k_arr_single, axis=0)[0:K_plot], color = "k", label="simulations (single source)" )
# plt.vlines(k_pred,0,1, label="Mean Passage Time", color="gray", linestyle="--")
# plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds)),1), p_percolation_preds[:min(K_plot, len(p_percolation_preds))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability ({num_robot} agents)")


plt.figure()
plt.plot(np.arange(K_plot),(np.sum(np.mean(I_sim, axis=0),axis=1)/num_robot)[0:K_plot], "k", label="simulation")
# plt.plot(np.arange(K_plot), x[0:K_plot],"--", color = "blue", label="Mean-Field-Approx.")            
# plt.plot(np.arange(K_plot),(np.sum(I,axis=1)/num_robot)[0:K_plot],"--", label="MFA (markov)")
plt.plot(np.arange(K_plot),percolation_percent_expected[0:K_plot],"--", color="red", label="Full Absorbing Model")
plt.plot(np.arange(K_plot),percolation_percent_lumped[0:K_plot],"--", color="green", label="Lumped Absorbing Model")

plt.legend()
plt.title(f"informed percentage({num_robot} agents)")