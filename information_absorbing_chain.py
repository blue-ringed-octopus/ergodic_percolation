# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:19:06 2026

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

num_regions = 5
num_robot = 4
num_states = int(sum([math.comb(num_regions+i-1, i)*math.factorial(num_regions-1+i)/(math.factorial(i)*math.factorial(num_regions-1)) for i in range(num_robot)]))
num_trial = 1000
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

p = dirichlet.rvs(1*np.ones(num_regions))[0]

S2 = np.sum(p**2)

directed = True

if directed:
    g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
    while not nx.is_strongly_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
else:
    g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)

for i in range(num_regions):
    g.add_edge(i,i)
    
edges = list((g.edges))

# P = FMMC(p, edges, directed=directed)
P = REMC(p, edges)

p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
K=50

pk=[p0.copy()]
for k in range(K):
    pk.append(P@pk[k].copy())
    
#meeting probability prediction
state_comb =  list(product(range(num_regions), repeat=2))
m = len(state_comb)
P2 = np.zeros((m,m))
p02 =np.zeros(m)
for I in  range(m):
    i1,i2 = state_comb[I]
    p02[I] = p0[i1].copy()*p0[i2].copy()
    for J in range(m):
        j1,j2 = state_comb[J]
        P2[I,J] = P[i1,j1].copy()*P[i2,j2].copy()
        
Q = P2.copy()
q0 = p02.copy()

q1 = np.zeros(len(q0))
for I in  range(m):
    i,j = state_comb[I]
    if i==j:
        Q[:,I] = 0#np.nan
        Q[I,:] = 0#np.nan
        q0[I] = 0#np.nan
        q1[I] = 1
K=100    

M = np.zeros((num_states,num_states))
for k in range(K):
    for i in range(num_states):
        for j in range(num_states):
            k
#%% sim    
informed_percentage_single_source = np.ones((num_trial, K))
kc_single = np.ones(num_trial)*np.nan
rho = np.zeros((num_trial, K, num_regions))
for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    for k in range(K): 
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                info = state[i,:].copy() |  state[j,:].copy() 
                state[i,:] = info
                state[j,:] = info
        informed_percentage_single_source[trial,k] = np.sum(state[:,0])/num_robot

        if (np.prod(state[:,0])):
            kc_single[trial]= np.nanmin([k,kc_single[trial]])
            break
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]    
            # r[robot] = np.random.choice(range(num_regions), 1, p=pk[k+1])[0]

            rho[trial,k,r[robot]]+=1/num_robot
        # r[0] = np.random.choice(range(num_regions), 1, p=np.linalg.matrix_power(P, k+1)@I[0])
