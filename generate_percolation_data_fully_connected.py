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
import os

if os.path.isfile("percolation_trials_complete.pkl"):    
    with open("percolation_trials_complete.pkl", "rb") as file:
        graph_data = pickle.load(file)
else:
    graph_data=[]
    
num_trial = 1000

# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
def get_P():
    num_regions = np.random.randint(40,80)
    g = nx.complete_graph(num_regions)
    num_regions = len(g)

    P = np.zeros((num_regions, num_regions))
    rho_bar = dirichlet.rvs(np.ones(num_regions))[0]
    for i in range(num_regions):
        P[:, i] = rho_bar

    return P




# graph_ids = [np.random.randint(len(graph_types)) for _ in range(900)]

while len(graph_data)< 1000:
# for graph_id in graph_ids:
    graph_id = 5
    P = get_P()
    graph_dat = {}
    num_regions = len(P)
    p0 = dirichlet.rvs(1*np.ones(num_regions))[0]   
    graph_dat["type"] ="complete"
    graph_dat["P"] = P
    graph_dat["p0"] = p0
    graph_dat["num_robots"] = np.random.randint(3,num_regions//2)
    graph_data.append(graph_dat)
    
#%% simulation
K = 1000
for graph_dat in tqdm(graph_data):
    if "p percolation single" not in graph_dat.keys():
        P = graph_dat["P"]
        num_robot = graph_dat["num_robots"]
        p0 = graph_dat["p0"]
        
        num_regions = len(P)
        kc_single = np.ones(num_trial)*np.nan
        kc = np.ones(num_trial)*np.nan
        rho = np.zeros((num_trial, K, num_regions))
        I_sim = np.zeros((num_trial,K, num_regions))
        for trial in range(num_trial):
            state = np.eye(num_robot, dtype=bool)
            r = np.random.choice(range(num_regions), num_robot, p=p0)
            for k in range(K): 
                for robot in range(num_robot):
                    I_sim[trial,k,r[robot]]+=state[robot,0]
                for i,j in combinations(range(num_robot), 2): 
                    if r[i]==r[j]:
                        info = state[i,:].copy() |  state[j,:].copy() 
                        state[i,:] = info
                        state[j,:] = info
                    
                if (np.prod(state[:,0])):
                    kc_single[trial]= np.nanmin([k,kc_single[trial]])
                    
                if (np.prod(state)):
                    kc[trial]= np.nanmin([k,kc[trial]])
                    break
                
                for robot in range(num_robot):
                    r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]    
        
                    rho[trial,k,r[robot]]+=1/num_robot
        
        rho = np.mean(rho,axis=0)
        
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
        
    
        graph_dat["kc single"] = np.nanmean(kc_single)
        graph_dat["kc all"] = np.nanmean(kc)
        graph_dat["p percolation single"] = np.mean(k_arr_single, axis=0)
        graph_dat["p percolation all"] = np.mean(k_arr, axis=0)
        graph_dat["informed percentage"] = np.mean(np.sum(I_sim/num_robot,axis=1), axis=0)
    with open("percolation_trials_complete.pkl", "wb") as file: 
        pickle.dump(graph_data, file)
        
# with open("percolation_trials.pkl", "rb") as file:
#     graph_data_prior = pickle.load(file)
# graph_data = graph_data_prior + graph_data
# with open("percolation_trials.pkl", "wb") as file:
#     pickle.dump(graph_data, file)