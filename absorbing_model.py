# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:56:31 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from scipy.stats import dirichlet
import networkx as nx
from tqdm import tqdm

num_regions = 25# np.random.randint(80,100)
num_robot = 3# np.random.randint(10,30)
num_trial = 1000
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

 

directed = False

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
        
# g = nx.grid_graph((8,8), periodic = True)
# g = nx.wheel_graph(num_regions)
# g = nx.hexagonal_lattice_graph(5,6, periodic = True)
g = nx.cycle_graph(num_regions)
# g = nx.barbell_graph(8,3)
num_regions = len(g)
A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,3)], axis=0).astype(bool)

P = np.zeros((num_regions, num_regions))
for i in range(num_regions):
   idx =  np.where(A[:,i])[0]
   P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]

# p_infty = np.linalg.matrix_power(P,9999)[:,0]
# p0=p_infty
p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
#%%
lambda_, v = np.linalg.eig(P)
idx = np.argsort(np.abs(lambda_))
lambda2 = np.abs(lambda_[idx[-2]])

P_infty = np.linalg.matrix_power(P, 9999)
P_perp = P-P_infty
P_perp_k = [np.linalg.matrix_power(P_perp,k) for k in range(500)]
Z_k = [np.sum(P_perp_k[:k+1], axis=0) for k in range(499)]
Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
kemeny = np.trace(Z)
kemeny_k = np.array([np.trace(x) for x in Z_k])
# kemeny_k = np.array([np.max(np.diag(x)) for x in Z_k])*num_regions
# kemeny_k = np.array([np.quantile(np.diag(x), 0.9) for x in Z_k])*num_regions
# kemeny_k = np.array([np.mean(np.diag(x))+1*np.std(np.diag(x)) for x in Z_k])*num_regions
eig_perp = np.linalg.eigvals(Z)
# test = np.array([np.linalg.norm(x) for x in Z_k])
plt.plot(num_regions/kemeny_k)


# test = lambda_
# for i in tqdm(np.arange(0,num_robot-1)):
#     test = np.kron(test,lambda_)
    
# test[1-np.abs(test)<0.0001] = 0
# kemeny_k_test = [sum([(1-l**k)/(1-l) for l in test]) for k in np.arange(1,10)]
p_percolation_preds = np.ones(1000)
P_kron=np.kron(P,P)
p0_kron = np.kron(p0, p0)
Q = P_kron.copy()
idx = [num_regions*i+i for i in range(num_regions)]
Q[idx,:] = 0 
Q[:,idx] = 0 
q0 = p0_kron.copy()
q0[idx] = 0 

q = [q0]
for k in range(5000):
    q.append(Q@q[-1].copy())

p_not_met = np.sum(q, axis=1)



#%% absorbing model


tau = 1/(1-lambda2)
# tau = kemeny/num_regions
# tau=kemeny
pk=[p0.copy()]
for k in range(5000):
    pk.append(P@pk[k].copy())
    
def Mk(k, pk):
    M = np.zeros((num_robot,num_robot))
    # if k==0:
    #     p_not_meet_conditional = 1-np.inner(p0,p0)
    # else:
    #     p = pk[k-1]
    #     # p_not_meet_conditional = sum([p[kk]/(1-p[kk])*sum([p[ii]*sum([P[jj,ii]*(1-P[jj,kk])for jj in range(num_regions)]) for ii in range(num_regions) if not ii==kk]) for kk in range(num_regions)])
    # p = pk[k]
    #     # p_not_meet_conditional = (1-(np.inner(p,p)))
    #     p_not_meet_conditional = 1- p_meet_conditional[k]
    for i in range(num_robot):
        for j in range(num_robot):
            I = j+1 #informed robot
            n = num_robot - I # number of uninformed robot
            if n==0:
                if i>=j:
                    M[i,j] = 1
            else:

                if k==0:
                    p_informed_condition = np.inner(p0,p0)
                else:
                    # p_not_met_k = p_not_met[k].copy()**(I+(1-I)*(lambda2)**(k))#**(I+(1-I)*np.exp(-k/tau))
                    # p_not_met_k1 = p_not_met[k-1].copy()**(I+(1-I)*(lambda2)**(k-1))#**(I+(1-I)*np.exp(-(k-1)/tau))
                    p_not_met_k = (1-lambda2**k)*p_not_met[k]**(I)+lambda2**k*p_not_met[k]
                    p_not_met_k1 = (1-lambda2**(k-1))*p_not_met[k-1]**(I)+lambda2**(k-1)*p_not_met[k-1]
                    p_met_k = 1-p_not_met_k
                    p_met_k1 = 1-p_not_met_k1
                    p_first_meet_k = p_met_k - p_met_k1
                    
                    p_informed_condition = p_first_meet_k/(p_not_met_k1)
                    # p_informed_condition =  1-(1-np.inner(pk[k], pk[k]))**(I+(1-I)*np.exp(-k/tau))
                # p_informed_condition = (num_regions/kemeny)*np.tanh(2*num_regions/kemeny)*(1-(p_not_meet_conditional)**I)
                # p_informed_condition = (num_regions/kemeny_k[k])*np.tanh(2*num_regions/kemeny_k[k])*(1-(p_not_meet_conditional)**I)
                # p_informed_condition = (1-(p_not_meet_conditional)**(I*((num_regions)/(kemeny))))
                # p_informed_condition = (1-(p_not_meet_conditional)**(I*((num_regions)*np.tanh(2*num_regions/kemeny_k[k])/kemeny_k[k])))
                # p_informed_condition = (1-(p_not_meet_conditional)**(I*((num_regions)*np.tanh(I*num_regions/kemeny_k[k])/kemeny_k[k])))
                # p_informed_condition = (1-(p_not_meet_conditional)**(I*(num_regions/kemeny)))
                # p_informed_condition = (1-(p_not_meet_conditional)**(I))

                # p_informed_condition = (1-lambda2)*(1-(p_not_meet_conditional)**I)
                # p_informed_condition = (1/(1+lambda2))*(1-(p_not_meet_conditional)**I)
                if i>=j:
                    r = i - j  #increment of informed robot 
                    M[i,j] = math.comb(n,r)*(p_informed_condition)**r*(1-p_informed_condition)**(n-r)
    return M

def p_absorb_markov(P, p0):
    K0 = 1000
    M0 = np.zeros((K0,num_robot,num_robot))
    
    v = np.zeros(num_robot-1)
    v[0]=1
    p_percolation_preds=[]
    prod = np.eye(num_robot-1)   
    
    for k in tqdm(range(K0)):
        M0[k,:,:] = Mk(k, pk)
                
    
    for T in np.arange(0,K0,1):
        prod = M0[T,:num_robot-1, :num_robot-1].copy()@prod
        p_percolation_preds.append(1- sum(prod@v))    
    k_pred =sum([(p_percolation_preds[k]-p_percolation_preds[k-1])*k for k in np.arange(1,K0)])
    
    M = M0
    # # k_pred = K0
    return p_percolation_preds, k_pred, M

# p_percolation_preds_fc , _ = p_absorb_fully_connected()
p_percolation_preds_single, k_pred, M = p_absorb_markov(P.copy(), p0.copy())
K = min(int(np.floor(3*k_pred)), len(M))
percolation_percent_expected =[1/num_robot]
v=np.zeros(num_robot)
v[0]=1
for k in range(K):
    v=M[k,:,:]@v
    percolation_percent_expected.append(1/num_robot*np.inner(v,np.arange(1,num_robot+1)))


p_percolation_preds = np.array([p**(num_robot-1) for p in p_percolation_preds_single])
#%% mean field model
    
    
# Ik = p0.copy()
Ik = np.zeros(num_regions)
Ik[np.argmin(p0)]=1
Uk = p0.copy()*(num_robot-1)

I=[Ik.copy()]
U=[Uk.copy()]
deltaI = []
for k in range(K):
    NI = sum(Ik)
    NU = sum(Uk)
    deltaI.append(np.array([Uk[i]*(1-(1-Ik[i]/NI)**(NI)) for i in range(num_regions)]))
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
kc = np.ones(num_trial)*np.nan
rho = np.zeros((num_trial, K, num_regions))
I_sim = np.zeros((num_trial,K+1, num_regions))
for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    I_sim[trial,0,r[0]] += 1
    for k in range(K): 
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                info = state[i,:].copy() |  state[j,:].copy() 
                state[i,:] = info
                state[j,:] = info
        # for region in range(num_regions):
        #     if region in r:
        #         robots = np.where(r==region)[0]
        #         state[robots,:] = np.sum([state[rob] for rob in robots], axis=0).astype(bool)
        for robot in range(num_robot):
            I_sim[trial,k+1,r[robot]]+=state[robot,0]
            
        if (np.prod(state[:,0])):
            kc_single[trial]= np.nanmin([k,kc_single[trial]])
            
        if (np.prod(state)):
            I_sim[trial,k+1:,:]= I_sim[trial,k+1,:]
            kc[trial]= np.nanmin([k,kc[trial]])
            break
        
        for robot in range(num_robot):
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

k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)  

k_arr_single = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc_single[i]):
        k[int(kc_single[i]):]=1
    k_arr_single.append(k)    

p_percolation_single_sim = np.mean(k_arr_single,axis=0)

K_plot = min(K, I_sim.shape[1])
# K_plot = 100

plt.figure()
plt.plot(np.arange(K_plot),(np.sum(np.mean(I_sim, axis=0),axis=1)/num_robot)[0:K_plot], "k", label="simulation")
# plt.plot(np.arange(K_plot), x[0:K_plot],"--", color = "blue", label="Mean-Field-Approx.")            
plt.plot(np.arange(K_plot),(np.sum(I,axis=1)/num_robot)[0:K_plot],"--", label="Mean-Field (Adjusted)")
plt.plot(np.arange(K_plot),percolation_percent_expected[0:K_plot],"--", color="red", label="Absorbing Model")
plt.legend()
plt.title(f"informed percentage({num_robot} agents, {num_regions} regions)")


plt.figure(dpi=800)
plt.plot(np.arange(0,K_plot,1), p_percolation_single_sim[0:K_plot], color = "k", label="simulations (single source)" )
# plt.vlines(k_pred,0,1, label="Mean Passage Time", color="gray", linestyle="--")
plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds_single)),1), p_percolation_preds_single[:min(K_plot, len(p_percolation_preds_single))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability (single source, {num_robot} agents, {num_regions} regions)")
plt.legend()
plt.figure(dpi=800)
plt.plot(np.arange(0,K_plot,1), np.mean(k_arr, axis=0)[0:K_plot], color = "k", label="simulations (single source)" )
# plt.vlines(k_pred,0,1, label="Mean Passage Time", color="gray", linestyle="--")
plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds)),1), p_percolation_preds[:min(K_plot, len(p_percolation_preds_single))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability (all source, {num_robot} agents, {num_regions} regions)")
plt.legend()

p_percolation_k = np.array([sum(kc_single==i) for i in range(K)])/num_trial
kc_sim = sum([k*p_percolation_k[k] for k in range(len(p_percolation_k))])
print(k_pred,kc_sim)

#%%
# import pickle 
# with open('data.pkl', 'rb') as file:
#     dat = pickle.load(file)
# dat["k"].append(num_regions/kemeny)
# dat["err"].append((k_pred - np.nanmean(kc_single))/np.nanmean(kc_single))
# with open("data.pkl", "wb") as file:
#     pickle.dump(dat, file)
    
# plt.figure()
# plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# plt.plot(dat["k"], dat["err"], ".")
# plt.xlabel("Normalized Kemeny Factor")
# plt.ylabel(r"$(K_{c,pred}-K_c)/K_c$")
