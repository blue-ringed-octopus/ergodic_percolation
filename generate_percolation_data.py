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

num_trial = 1000

# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
graph_types = ["er-directed", "er-undirected", "cycle", "barbell", "grid"]

def get_P(graph_id):
    match graph_id:
        case 0:
            num_regions = np.random.randint(40,80)
            p_edge = 2*np.log(num_regions)/num_regions
            g = nx.erdos_renyi_graph(num_regions, p = p_edge , directed = True)
            while not nx.is_strongly_connected(g):
                g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = True)
        case 1:
            num_regions = np.random.randint(40,80)
            p_edge = np.log(num_regions)/num_regions
            g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = False)
            while not nx.is_connected(g):
                g = nx.erdos_renyi_graph(num_regions, p = p_edge, directed = False)
        case 2:
            g = nx.cycle_graph(np.random.randint(40,80))
        case 3:
            bell = np.random.randint(20,40)
            bar = np.random.randint(1,4)
            g = nx.barbell_graph(bell,bar)
            
        case 4:
            w = np.random.randint(4,9)
            h = np.random.randint(4,9)
            g = nx.grid_graph((h,w), periodic = True)
    num_regions = len(g)
    degree = np.random.randint(2,5)        
    A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
    A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,degree)], axis=0).astype(bool)
    P = np.zeros((num_regions, num_regions))

    for i in range(num_regions):
        idx =  np.where(A[:,i])[0]
        P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]

    return P




graph_data = []
graph_ids = [np.random.randint(len(graph_types)) for _ in range(100)]

for graph_id in graph_ids:
    P = get_P(graph_id)
    graph_dat = {}
    num_regions = len(P)
    p0 = dirichlet.rvs(1*np.ones(num_regions))[0]   
    graph_dat["type"] = graph_types[graph_id]
    graph_dat["P"] = P
    graph_dat["p0"] = p0
    graph_dat["num_robots"] = np.random.randint(3,num_regions//2)

    graph_data.append(graph_dat)
    
#%% simulation
K = 1000
for graph_dat in graph_data:
    P = graph_dat["P"]
    num_robot = graph_dat["num_robots"]
    p0 = graph_dat["p0"]
    
    num_regions = len(P)
    kc_single = np.ones(num_trial)*np.nan
    kc = np.ones(num_trial)*np.nan
    rho = np.zeros((num_trial, K, num_regions))
    I_sim = np.zeros((num_trial,K, num_regions))
    for trial in tqdm(range(num_trial)):
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


with open("percolation_trials.pkl", "wb") as file:
    pickle.dump(graph_data, file)