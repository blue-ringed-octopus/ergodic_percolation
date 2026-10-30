# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:17:46 2026

@author: hibado
"""

from simulation import World, Agent
import matplotlib
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np
import networkx as nx
import pickle

num_trial = 100
K = 10000

h = 20
w = 20
num_regions = int(h*w)

radius = 5

# density = 0.5
# num_agents = int(num_regions*density)
num_agents = np.arange(10, 1000, 10)
p0 = np.ones(num_regions)/num_regions
g = nx.grid_graph((h,w), periodic = True)
# degree = int(h*w/radius)

A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,radius)], axis=0).astype(bool)
P = np.zeros(A.shape)
for i in range(num_regions):
    idx =  np.where(A[:,i])[0]
    P[idx, i] = np.ones(len(idx))/len(idx)
std_arr = []
mean_arr = []
data={}    
#%%
k_arrs=[]
# kc = np.ones((len(num_agents),num_trial))*np.nan
for n in tqdm(num_agents):
    kc = np.ones(num_trial)*np.nan
    agents = {}
    for i in range(n):
        agents[i] = Agent(i)
    world = World("grid",agents)
    for trial in range(num_trial):
        world.reset(p0=p0, P=P)
        for k in range(K):
            information_state = world.step()
            if (np.prod(information_state)):
                kc[trial] = np.nanmin([k,kc[trial]])
                break
    k_arr = []
    for i in range(num_trial):
        k = np.zeros(K)
        if not np.isnan(kc[i]):
            k[int(kc[i]):]=1
        k_arr.append(k)  
    k_arrs.append(k_arr)
    std_arr.append(np.std(kc))
    mean_arr.append(np.mean(kc))
    data[num_agents] = kc.copy()
    plt.figure()
    plt.plot(np.mean(k_arr, axis=0)[:int(np.mean(kc)*2)])
    plt.vlines(np.mean(kc),0,1, linestyle= "--")
    plt.title(f"num agents: {n}, std: {np.std(kc)}")
    plt.show()
with open("phase_transition_dat.pkl", "wb") as file:
    pickle.dump(data, file)
#%%
fontsize = 12
kc0 = np.argmax(np.diff(np.mean(k_arrs[0], axis=0)))
plt.figure(dpi=800)
plt.plot(np.mean(k_arrs[-1], axis=0)[0:2*kc0], linewidth=5, label=f"N={num_agents[-1]}")
plt.plot(np.mean(k_arrs[10], axis=0)[0:2*kc0], linewidth=5, label=f"N={num_agents[10]}")
plt.plot(np.mean(k_arrs[1], axis=0)[0:2*kc0], linewidth=5, label=f"N={num_agents[1]}")
plt.plot(np.mean(k_arrs[0], axis=0)[0:2*kc0], linewidth=5, label=f"N={num_agents[0]}")
plt.title(f"Probability of Full Percolation \n (1000 trials) ", fontsize=fontsize, weight="bold")
plt.ylabel("Pr(Full Percolation)", weight="bold", fontsize=fontsize)
plt.xlabel("Timestep (k)", weight="bold", fontsize=fontsize)
plt.legend(loc="best", prop={'weight':"bold"})
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

plt.figure(dpi=800)
plt.plot(num_agents, std_arr, linewidth=5)
plt.title("Standard Deviation of Full Percolation Time \n vs Number of Agents \n (1000 trials)", fontsize=fontsize, weight="bold")
plt.ylabel("Standard Deviation ", weight="bold", fontsize=fontsize)
plt.xlabel("Number of Agents", weight="bold", fontsize=fontsize)
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")
plt.figure(dpi=800)
plt.plot(num_agents, mean_arr, linewidth=5)
plt.title("Mean Full Percolation Time vs Number of Agents \n (1000 trials) ", fontsize=fontsize, weight="bold")
plt.ylabel("Mean Full Percolation Time ", weight="bold", fontsize=fontsize)
plt.xlabel("Number of Agents", weight="bold", fontsize=fontsize)
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")