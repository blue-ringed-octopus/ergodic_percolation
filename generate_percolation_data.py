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

with open("resources/percolation_trials.pkl", "rb") as file:
    graph_data = pickle.load(file)
    
num_trial = 10000

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
    degree = np.random.randint(2,10)        
    A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
    A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,degree)], axis=0).astype(bool)
    P = np.zeros((num_regions, num_regions))

    for i in range(num_regions):
        idx =  np.where(A[:,i])[0]
        P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]

    return P




# graph_ids = [np.random.randint(len(graph_types)) for _ in range(900)]

while len(graph_data)< 1000:
# for graph_id in graph_ids:
    graph_id = np.random.randint(len(graph_types))
    P = get_P(graph_id)
    graph_dat = {}
    num_regions = len(P)
    p0 = dirichlet.rvs(1*np.ones(num_regions))[0]   
    graph_dat["type"] = graph_types[graph_id]
    graph_dat["P"] = P
    graph_dat["p0"] = p0
    graph_dat["num_robots"] = np.random.randint(3,num_regions//10)
    graph_data.append(graph_dat)
    
#%% simulation
from simulation import World, Agent, run_trial
K = 9000
for graph_dat in tqdm(graph_data):
    if True:#"p percolation single" not in graph_dat.keys():
        P = graph_dat["P"]
        num_robot = graph_dat["num_robots"]
        p0 = graph_dat["p0"]
        
        num_regions = len(P)
        kc_single = np.ones(num_trial)*np.nan
        kc = np.ones(num_trial)*np.nan
        I_sim = np.zeros((num_trial,K))
        agents = {}
        for i in range(num_robot):
            agents[i] = Agent(i)
        world = World(graph_dat["type"],agents)
        kc_single, kc, I_sim = run_trial(world,P,p0, num_trial, K)
        
        
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
        graph_dat["informed percentage"] = {"mean": np.mean(I_sim, axis=0), 
                                            "Q0": np.quantile(I_sim, 0, axis=0),
                                            "Q1": np.quantile(I_sim, 0.25, axis=0),
                                            "Q2": np.quantile(I_sim, 0.5, axis=0),
                                            "Q3": np.quantile(I_sim, 0.75, axis=0),
                                            "Q4": np.quantile(I_sim, 1, axis=0)}
        
    # with open("percolation_trials.pkl", "wb") as file: 
    #     pickle.dump(graph_data, file)
        
# with open("percolation_trials.pkl", "rb") as file:
#     graph_data_prior = pickle.load(file)
# graph_data = graph_data_prior + graph_data
# with open("percolation_trials.pkl", "wb") as file:
#     pickle.dump(graph_data, file)