# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 13:49:31 2026

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

num_regions = 200
num_robot = 50
num_trial = 1000
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

 

directed = True

if directed:
    p_edge = 2*np.log(num_regions)/num_regions
    # p_edge = 1
    g = nx.erdos_renyi_graph(num_regions, p = p_edge , directed = True)
    while not nx.is_strongly_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = True)
else:
    p_edge = np.log(num_regions)/num_regions
    g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = False)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = False)

    
A = nx.adjacency_matrix(g).toarray().T+np.eye(num_regions)
#%%
P = np.zeros((num_regions, num_regions))
for i in range(num_regions):
   idx =  np.where(A[:,i] == 1)[0]
   P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]
# p = dirichlet.rvs(1*np.ones(num_regions))[0]

# if p_edge == 1:
#     P = np.outer(p, np.ones(num_regions))
# else:
#     P = FMMC(p, A, reversible = False)
# # P = REMC(p, edges)

p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
# p0 = p
# p0 = np.ones(num_regions)/num_regions 
lambda_, _ = np.linalg.eig(P)
lambda_ = np.sort(np.abs(lambda_))
lambda2 = lambda_[-2]

#%% absorbing model
def p_absorb_markov(P, p0):
    K0 = 50
    M0 = np.zeros((K0,num_robot,num_robot))
    pk=[p0.copy()]
    
    for k in range(K0):
        pk.append(P@pk[k].copy())
        
    for k in tqdm(range(K0)):
        if k==0:
            p_not_meet_conditional = 1-sum([p0[i]**2 for i in range(num_regions)])
        else:
            p = pk[k-1]
            p_not_meet_conditional = sum([p[kk]/(1-p[kk])*sum([p[ii]*sum([P[jj,ii]*(1-P[jj,kk])for jj in range(num_regions)]) for ii in range(num_regions) if not ii==kk]) for kk in range(num_regions)])

        for i in range(num_robot):
            for j in range(num_robot):
                I = j+1 #informed robot
                n = num_robot - I # number of uninformed robot
                if n==0:
                    if i>=j:
                        M0[k,i,j] = 1
                else:
                    r = i - j  #increment of informed robot 
                    # p_informed = 1-(1-p_meeting_conditional[k])**I
                    # Ik = pk[k]*I
                    # p_informed = np.sum([1-(1-Ik[ii]/I)**I for ii in range(num_regions)])
                    # p_informed_condition =  p_informed/(1-sum(p_informed[0:k]))
                    # p_informed_condition = 1/(1+lambda2)*np.sum( [pk[k][ii]*(1-(1-Ik[ii]/I)**I) for ii in range(num_regions)])
                    p_informed_condition = 1- p_not_meet_conditional**I
                    # # p_informed_condition = np.sum( [1/(n)*(num_robot*pk[k][ii]-Ik[ii])*(1-(1-Ik[ii]/I)**I) for ii in range(num_regions)])

                    if i>=j:
                        M0[k,i,j] = math.comb(n,r)*(p_informed_condition)**r*(1-p_informed_condition)**(n-r)
                
    v = np.zeros(num_robot-1)
    v[0]=1
    # v = [ sum([p0[i]]) for n in np.arange(1,num_robot-1)]
    p_percolation_preds=[]
    prod = np.eye(num_robot-1)
    for T in np.arange(0,K0,1):
        prod = M0[T,:num_robot-1, :num_robot-1].copy()@prod
        p_percolation_preds.append(1- sum(prod@v))    
    k_pred = sum([(p_percolation_preds[k]-p_percolation_preds[k-1])*k for k in np.arange(1,K0)])
    
    if 2*k_pred>K0:
        K = int(np.ceil(2*k_pred))
        M = np.zeros((K,num_robot,num_robot))
        M[0:K0,:,:] = M0
        for k in np.arange(K0, K):
            pk.append(P@pk[k].copy())
        
        for k in np.arange(K0, K):
            p = pk[k-1]
            p_not_meet_conditional = sum([p[kk]/(1-p[kk])*sum([p[ii]*sum([P[jj,ii]*(1-P[jj,kk])for jj in range(num_regions)]) for ii in range(num_regions) if not ii==kk]) for kk in range(num_regions)])

            for i in range(num_robot):
                for j in range(num_robot):
                    I = j+1 #informed robot
                    n = num_robot - I # number of uninformed robot
                    if n==0:
                        if i>=j:
                            M[k,i,j] = 1
                    else:
                        r = i - j  #increment of informed robot 
                        
                        p_informed_condition = 1- p_not_meet_conditional**I
                        if i>=j:
                            M[k,i,j] = math.comb(n,r)*(p_informed_condition)**r*(1-p_informed_condition)**(n-r)
        for T in np.arange(K0, K,1):
            prod = M[T,:num_robot-1, :num_robot-1].copy()@prod
            p_percolation_preds.append(1- sum(prod@v))    
        k_pred = sum([(p_percolation_preds[k]-p_percolation_preds[k-1])*k for k in np.arange(1,K0)])
    else:
        M = M0
        
    return p_percolation_preds, k_pred, M

# p_percolation_preds_fc , _ = p_absorb_fully_connected()
p_percolation_preds, k_pred, M = p_absorb_markov(P.copy(), p0.copy())
K = int(np.ceil(2*k_pred))
percolation_percent_expected =[1/num_robot]
v=np.zeros(num_robot)
v[0]=1
for k in range(K):
    v=M[k,:,:]@v
    percolation_percent_expected.append(1/num_robot*np.inner(v,np.arange(1,num_robot+1)))

# if np.ceil(k_pred*2)>K:
#     K = int(np.ceil(k_pred*2))
#     p_percolation_preds, k_pred, M = p_absorb_markov(P.copy(), p0.copy())
#     percolation_percent_expected =[1/num_robot]
#     v=np.zeros(num_robot)
#     v[0]=1
#     for k in range(K):
#         v=M[k,:,:]@v
#         percolation_percent_expected.append(1/num_robot*np.inner(v,np.arange(1,num_robot+1)))

#%% mean field model
pk=[p0.copy()]
for k in range(K):
    pk.append(P@pk[k].copy())
    
    
Ik = p0.copy()
Uk = p0.copy()*(num_robot-1)

I=[Ik.copy()]
U=[Uk.copy()]
deltaI = []
for k in range(K):
    NI = sum(Ik)
    NU = sum(Uk)
    deltaI.append(1/(1+lambda2)*np.array([Uk[i]*(1-(1-Ik[i]/NI)**NI) for i in range(num_regions)]))
    # UI = (num_robot-NI)*pk[k]
    # UI = (num_robot)*pk[k] - Ik
    Ik = P@(Ik + deltaI[-1])
    Uk = P@(Uk - deltaI[-1])
    I.append(Ik.copy())
    U.append(Uk.copy())
    # print(Ik[i]/NI, pk[i])
I = np.array(I)
#%% simulation
kc_single = np.ones(num_trial)*np.nan
rho = np.zeros((num_trial, K, num_regions))
I_sim = np.zeros((num_trial,K, num_regions))
for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=p0)
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
            # r[robot] = np.random.choice(range(num_regions), 1, p=pk[k+1])[0]

            rho[trial,k,r[robot]]+=1/num_robot
        # r[0] = np.random.choice(range(num_regions), 1, p=np.linalg.matrix_power(P, k+1)@I[0])

rho = np.mean(rho,axis=0)
#%%
x0 = 1/num_robot
x=[x0]
for k in range(K):
    dx=0
    xk = x[-1]
    for pi in pk[k]:
        dx+= (1-(1-pi)**(xk*num_robot))*pi
    dx*=(1-xk)
        
    x.append((xk+dx))
    

# p_informed_sim = np.sum(np.mean(I_sim, axis=0),axis=1)/num_robot
# p_informed = np.sum(I,axis=1)/num_robot

# p_first_informed_sim = np.concatenate([[p_informed_sim[0]], np.diff(p_informed_sim)])
# p_first_informed = np.concatenate([[p_informed[0]], np.diff(p_informed)])

# p_informed_conditional_sim = [p_first_informed_sim[k]/(1-sum(p_first_informed_sim[:k])) for k in range(K)]
# p_informed_conditional = [p_first_informed[k]/(1-sum(p_first_informed[:k])) for k in range(K)]

# plt.figure()
# plt.plot(p_informed_conditional_sim, '.')
# plt.plot(p_informed_conditional)
# plt.title("Informed given not informed before k")


# region = 0
# plt.figure()
# plt.plot(np.mean(I_sim, axis=0)[:,region])
# plt.plot(I[:,region])
# plt.title(f"Informed robots in region {region} ")



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
plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds)),1)+1, p_percolation_preds[:min(K_plot, len(p_percolation_preds))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability ({num_robot} agents, {num_regions} regions)")


plt.legend()

plt.figure()
plt.plot(np.arange(K_plot)+1,(np.sum(np.mean(I_sim, axis=0),axis=1)/num_robot)[0:K_plot], "k", label="simulation")
plt.plot(np.arange(K_plot), x[0:K_plot],"--", color = "blue", label="Mean-Field-Approx.")            
# plt.plot(np.arange(K_plot),(np.sum(I,axis=1)/num_robot)[0:K_plot],"--", label="MFA (markov)")
plt.plot(np.arange(K_plot)+1,percolation_percent_expected[0:K_plot],"--", color="red", label="Absorbing Model")

plt.legend()
plt.title(f"informed percentage({num_robot} agents, {num_regions} regions)")
