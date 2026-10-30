# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:56:31 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import pickle 
from simulation import World, Agent
num_trial = 10

output_name = "resources/percolation_trials_pred_complete.pkl"
with open("resources/percolation_trials_complete.pkl", "rb") as file:
    graph_data = pickle.load(file)

dat = graph_data[0]
dat = graph_data[0]
P = dat["P"]
num_regions = len(P)
p0 = dat["p0"]
num_robot = dat["num_robots"]
K = 500

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

#%% sim2
agents = {}
for i in range(num_robot):
    agents[i] = Agent(i)
    world = World(dat["graph_type"],agents)
    
for trial in tqdm(range(num_trial)):
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
    