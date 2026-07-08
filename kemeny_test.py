# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 06:51:00 2026

@author: hibado
"""

import numpy as np
from scipy.stats import dirichlet
import networkx as nx
from tqdm import tqdm

num_regions = 2
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
        
g = nx.grid_graph((5,5), periodic = False)
# g = nx.wheel_graph(num_regions)
# g = nx.hexagonal_lattice_graph(5,10, periodic = True)
num_regions = len(g)
A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
# A = (A@A).astype(bool)

P = np.zeros((num_regions, num_regions))
for i in range(num_regions):
   idx =  np.where(A[:,i])[0]
   P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]
# P=np.array([[0.9, 0.9],
#             [0.1, 0.1]])
p_infty = np.linalg.matrix_power(P,9999)[:,0]
# p0=p_infty

lambda_, _ = np.linalg.eig(P)
idx = np.argsort(np.abs(lambda_))
lambda2 = np.abs(lambda_[idx[-2]])

Z = np.linalg.inv(np.eye(num_regions)-P+np.linalg.matrix_power(P, 9999))
kemeny = np.trace(Z)
#%%


M=np.ones((num_trial, num_regions))*np.nan
for trial in range(num_trial):
    r = np.random.randint(0,num_regions)
    for k in np.arange(1,9999):
        r =  np.random.choice(range(num_regions),  p=P[:,r])   
        M[trial,r] = np.nanmin([M[trial,r], k])
        if not np.isnan(M[trial,:]).any():
            break
        
mean_hitting = np.nanmean(M, axis=0)
kemeny_sim = np.inner(mean_hitting, p_infty)