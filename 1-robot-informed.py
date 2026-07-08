# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:17:02 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

from tqdm import tqdm

num_regions = 2
num_robot = 3
num_trial = 10000
K=10
rho0 = dirichlet.rvs(np.ones(num_regions)).T
P = dirichlet.rvs(np.ones(num_regions), 2).T
lambda_, _ = np.linalg.eig(P)
lambda_ = np.sort(np.abs(lambda_))
lambda2 = lambda_[-2]
rho0_uninformed = np.array([rho0[0]*rho0[1]**2, rho0[1]*rho0[0]**2])
rho0_combined = np.concatenate([rho0_uninformed, [1-sum(rho0_uninformed)]])

# rho0_uninformed /= np.sum(rho0_uninformed)

P_uninformed = np.array([[P[0,0]*P[1,1]**2,P[1,0]*P[0,1]**2],
                         [P[0,1]*P[1,0]**2,P[1,1]*P[0,0]**2]])
P_informed =  1- np.sum(P_uninformed[0:4,:], axis=0)

P_combined = np.eye(3)
P_combined[0:2,0:2] = P_uninformed
P_combined[2,0:2] = P_informed

rho = []
for k in range(K):
    rho.append(np.linalg.matrix_power(P_combined, k)@rho0_combined)
rho = np.array(rho)
p_informed = rho[:,-1]

P_informed_conditional = []
rho_test = []
for k in range(K):
    rho_test.append(np.linalg.matrix_power(P_uninformed,k)@rho0_uninformed)
    rho_test[-1] /=sum(rho_test[-1])
    p = P_informed@np.linalg.matrix_power(P_uninformed,k)@rho0_uninformed
    P_informed_conditional.append(p/(np.ones(2)@np.linalg.matrix_power(P_uninformed,k)@rho0_uninformed))
    
P_informed_conditional2 = []
rho = []
P_informed_marginal = []

for k in range(K):
    rho.append(np.linalg.matrix_power(P_combined,k)@rho0_combined)
    rho_normalized = rho[-1][0:2]/np.sum(rho[-1][0:2])
    P_informed_conditional2.append((P_informed@rho_normalized))
    P_informed_marginal.append(P_combined[-1,:]@rho[-1])
plt.figure()
plt.plot(P_informed_conditional2)
plt.plot(P_informed_conditional)

plt.figure()
plt.plot(P_informed_marginal)
plt.plot(p_informed)

#%% simulation
kc_single = np.ones(num_trial)*np.nan
rho = np.zeros((num_trial, K, num_regions))
I_sim = np.zeros((num_trial,K, num_regions))
for trial in tqdm(range(num_trial)):
    informed = False

    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=rho0.T[0])
    for k in range(K): 
        for robot in np.arange(1, num_robot):
            if r[0] == r[robot]:
                kc_single[trial] = k
                informed = True
        if informed:
            break
        
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]    

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
plt.plot(np.arange(0,min(K_plot, len(p_informed)),1), p_informed[:min(K_plot, len(p_informed))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability ({num_robot} agents)")


# plt.figure()
# plt.plot(np.arange(K_plot),(np.sum(np.mean(I_sim, axis=0),axis=1)/num_robot)[0:K_plot], "k", label="simulation")
# # plt.plot(np.arange(K_plot), x[0:K_plot],"--", color = "blue", label="Mean-Field-Approx.")            
# # plt.plot(np.arange(K_plot),(np.sum(I,axis=1)/num_robot)[0:K_plot],"--", label="MFA (markov)")
# plt.plot(np.arange(K_plot),percolation_percent_expected[0:K_plot],"--", color="red", label="Full Absorbing Model")
# plt.plot(np.arange(K_plot),percolation_percent_lumped[0:K_plot],"--", color="green", label="Lumped Absorbing Model")

# plt.legend()
# plt.title(f"informed percentage({num_robot} agents)")