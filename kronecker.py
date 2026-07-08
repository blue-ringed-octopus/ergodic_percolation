# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:02:27 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from scipy.stats import dirichlet
import networkx as nx
from tqdm import tqdm

num_regions = 20
num_robot = np.random.randint(10,30)
num_robot = 20
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
        
g = nx.grid_graph((3,3), periodic = True)
# g = nx.wheel_graph(num_regions)
# g = nx.hexagonal_lattice_graph(5,10, periodic = True)
num_regions = len(g)
A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
# A = (A@A).astype(bool)

P = np.zeros((num_regions, num_regions))
for i in range(num_regions):
   idx =  np.where(A[:,i])[0]
   P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]

# p_infty = np.linalg.matrix_power(P,9999)[:,0]
# p0=p_infty
p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
#%%
lambda_, _ = np.linalg.eig(P)
idx = np.argsort(np.abs(lambda_))
lambda2 = np.abs(lambda_[idx[-2]])

P_infty = np.linalg.matrix_power(P, 9999)
P_perp = P-P_infty
P_perp_k = [np.linalg.matrix_power(P_perp,k) for k in range(500)]
Z_k = [np.sum(P_perp_k[:k+1], axis=0) for k in range(499)]
Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
kemeny = np.trace(Z)
kemeny_k = np.array([np.trace(x) for x in Z_k])

eig_perp = np.linalg.eigvals(Z)
# test = np.array([np.linalg.norm(x) for x in Z_k])
plt.plot(num_regions/kemeny_k[0:20])
P2 = np.kron(P,P)
pi2 = np.kron(P_infty[:,0],P_infty[:,0])
P_perp2 = P2-np.outer(pi2,np.ones(num_regions**2))

Z2 = np.linalg.inv(np.eye(num_regions**2)-P_perp2)
kemeny2 = np.trace(Z2)


test = np.kron(lambda_,lambda_)
test[1-np.abs(test)<0.0001] = 0
kemeny2_test = sum([1/(1-l) for l in test])

idx = [num_regions*i+i for i in range(num_regions)]
C = np.zeros((num_regions**2-num_regions+1, num_regions**2))
C[0,idx]=1
j= 1 
for i in range(num_regions**2):
    if i not in idx:
        C[j,i] = 1
        j+=1
pi_tilde = pi2[idx]/sum(pi2[idx])
D = np.zeros(C.T.shape)
D[idx,0] = pi_tilde
j=1
for i in range(num_regions**2):
    if i not in idx:
        D[i,j] = 1
        j+=1
        
P_tilde = C@P2@D