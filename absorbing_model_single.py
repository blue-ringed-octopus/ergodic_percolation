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
import pickle 
from absorbing_model import Absoorbing_Model
from simulation import World, Agent, run_trial
num_trial = 1000
with open("resources/percolation_trials.pkl", "rb") as file:
    graph_data = pickle.load(file)

# with open("resources/percolation_trials_complete.pkl", "rb") as file:
#     graph_data = pickle.load(file)
        
        
dat = graph_data[3]
P = dat["P"]
num_regions = len(P)
p0 = dat["p0"]
num_robot = dat["num_robots"]
graph_type = dat["type"]

# num_regions = 50# np.random.randint(80,100)
# num_robot = 3# np.random.randint(10,30)
# P = np.tile(dirichlet.rvs(np.ones(num_regions)),(num_regions,1)).T
# p0 = dirichlet.rvs(np.ones(num_regions))[0]
# graph_type= "complete"
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

 

# directed = False

# if directed:
#     p_edge = 2*np.log(num_regions)/num_regions
#     # p_edge = 1
#     g = nx.erdos_renyi_graph(num_regions, p = p_edge , directed = True)
#     while not nx.is_strongly_connected(g):
#         g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = True)
# else:
#     p_edge = np.log(num_regions)/num_regions
#     g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = False)
#     while not nx.is_connected(g):
#         g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = False)
        
# # g = nx.grid_graph((8,8), periodic = True)
# # g = nx.wheel_graph(num_regions)
# # g = nx.hexagonal_lattice_graph(5,6, periodic = True)
# g = nx.cycle_graph(num_regions)
# # g = nx.barbell_graph(8,3)
# num_regions = len(g)
# A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
# A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,3)], axis=0).astype(bool)

# P = np.zeros((num_regions, num_regions))
# for i in range(num_regions):
#    idx =  np.where(A[:,i])[0]
#    P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]

# # p_infty = np.linalg.matrix_power(P,9999)[:,0]
# # p0=p_infty
# p0 = dirichlet.rvs(1*np.ones(num_regions))[0]

#%% absorbing model
model = Absoorbing_Model(P, p0, num_robot)

p_percolation_preds_single ,k_pred, M =model.pred(12)
p_not_met = model.p_not_met

# p_percolation_preds_fc , _ = p_absorb_fully_connected()
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

agents = {}
for i in range(num_robot):
    agents[i] = Agent(i)
world = World(graph_type,agents)
kc_single, kc, I_sim = run_trial(world,P,p0, num_trial, 100000)
# for trial in tqdm(range(num_trial)):
#     world.reset(p0=p0, P=P)
#     for k in range(K):
#         I_sim[trial, k] = world.get_informed_percentage(0)
#         information_state = world.step()
#         if (np.prod(information_state[:,0])):
#             kc_single[trial]= np.nanmin([k,kc_single[trial]])
            
#         if (np.prod(information_state)):
#             kc[trial] = np.nanmin([k,kc[trial]])
#             I_sim[trial, k+1:] = world.get_informed_percentage(0)
#             break
#%%
pk = model.pk.copy()
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

K_plot = np.min((K, I_sim.shape[1], len(p_percolation_preds_single)))
# K_plot = 500

plt.figure()
plt.plot(np.arange(K_plot),np.mean(I_sim, axis=0)[0:K_plot], "k", label="simulation")
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
# plt.plot(dat["p percolation single"][1:K_plot],'--', label="saved")
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

p_percolation_k = np.array([sum(kc_single==i) for i in range(1+int(max(kc_single)))])/num_trial
kc_sim = np.inner(p_percolation_k, np.arange(0,len(p_percolation_k)))#sum([k*p_percolation_k[k] for k in range(len(p_percolation_k))])
print(k_pred,kc_sim)
