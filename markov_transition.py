# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 19:48:23 2025

@author: hibado
"""


import numpy as np
import matplotlib.pyplot as plt 
from itertools import combinations, product
# from graph_planner import REMC, FMMC
import networkx as nx
from scipy.stats import dirichlet
import math 
from tqdm import tqdm

np.set_printoptions(precision=2)

num_regions = 100
num_robot = 50
num_trial = 1000
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
p = np.ones(num_regions)
p = p/sum(p)
# p = dirichlet.rvs(1*np.ones(num_regions))[0]

S2 = np.sum(p**2)

directed = False

if directed:
    g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
    while not nx.is_strongly_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
else:
    g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)

# g = nx.erdos_renyi_graph(num_regions, 1, directed = directed)

for i in range(num_regions):
    g.add_edge(i,i)
    
edges = list((g.edges))
A = nx.adjacency_matrix(g).toarray().T+np.eye(num_regions)

# P = FMMC(p, edges, directed=directed)
# P = REMC(p, edges)
P = np.zeros((num_regions, num_regions))
for i in range(num_regions):
   idx =  np.where(A[:,i] == 1)[0]
   P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]
   
lambda_, _ = np.linalg.eig(P)
lambda_ = np.sort(np.abs(lambda_))
lambda2 = lambda_[-2]

p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
# p0 = p


# def p_absorb_fully_connected():
#     M = np.zeros((num_robot,num_robot))
#     for i in range(num_robot):
#         for j in range(num_robot):
#             #j: informed 
#             I = j+1
#             n = num_robot - I # number of uninformed robot
#             r = i - j  #increment of informed robot 
#             if i>=j:
#                 M[i,j] = math.comb(n,r)*(1-(1-S2)**I)**r*((1-S2)**I)**(n-r)
                            
#     Q = M[:num_robot-1, :num_robot-1]
#     v = np.zeros(num_robot-1)
#     v[0]=1
#     k_pred = np.ones(num_robot-1)@np.linalg.inv(np.eye(num_robot-1)-Q)@v
#     K = int(2.5*k_pred)
#     p_percolation_preds=[]
#     for T in np.arange(1,K+1,1):
#         p_percolation_preds.append(1- sum(np.linalg.matrix_power(Q, T)@v))    
#     return p_percolation_preds, k_pred-1


def p_absorb_markov(P, p0):
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
        else:
            q1[I] = 1
    K=50    
    # absortion time prediction (markov region)
    p_not_meeting=[1-np.inner(p0,p0)]
    p_not_meeting_conditional=[p_not_meeting[-1]]
    # Q_k = np.eye(m)
    pk= p02 
    for T in np.arange(1,K+1,1):
        # Q_k = Q@Q_k
        # p_not_meeting.append(q1.T@P2@Q_k@q0)
        # p_not_meeting_conditional.append(p_not_meeting[-1]/(q1.T@Q_k@q0))
        p_not_meeting.append(q1.T@P2@(q1*pk))
        p_not_meeting_conditional.append(p_not_meeting[-1]/(q1.T@(pk)))
        # Q_k = Q@Q_k
        pk = P2@p02
        
    # p_meeting_conditional = [p_meeting[k]/(1-sum(p_meeting[:k])) for k in range(len(p_meeting))]
    # p_meeting_conditional = p_meeting
    M = np.zeros((K,num_robot,num_robot))
    for k in tqdm(range(K)):
        for i in range(num_robot):
            for j in range(num_robot):
                I = j+1 #informed robot
                n = num_robot - I # number of uninformed robot
                r = i - j  #increment of informed robot 
                p_informed = (1-(p_not_meeting_conditional[k])**I)

                if i>=j:
                    M[k,i,j] = math.comb(n,r)*(p_informed)**r*(1-p_informed)**(n-r)
                
    v = np.zeros(num_robot-1)
    v[0]=1
    
    p_percolation_preds=[]
    prod = np.eye(num_robot-1)
    for T in np.arange(0,K,1):
        prod = M[T,:num_robot-1, :num_robot-1].copy()@prod
        p_percolation_preds.append(1- sum(prod@v))    
    k_pred = np.ones(num_robot-1)@np.linalg.inv(np.eye(num_robot-1)-M[0,:num_robot-1, :num_robot-1])@v
    return p_percolation_preds, k_pred

# p_percolation_preds_fc , _ = p_absorb_fully_connected()
p_percolation_preds, k_pred = p_absorb_markov(P.copy(), p0.copy())
K = int(3*k_pred)


#%%
kc = np.ones(num_trial)*np.nan
kc_single = np.ones(num_trial)*np.nan

informed_percentage_single_source = np.ones((num_trial, K))
informed_percentage = np.ones((num_trial, K))

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
        informed_percentage[trial,k] = np.min(np.sum(state, axis=1)/num_robot)
        
        if (np.prod(state[:,0])):
            kc_single[trial]= np.nanmin([k,kc_single[trial]])
        if (np.prod(state)):
            kc[trial]= k
            break
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])
#%%
K=50
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
        


plt.figure(dpi=800)
# plt.plot(np.arange(0,K,1), np.mean(k_arr, axis=0), color = "red", label="simulations (multi source)" )
plt.plot(np.arange(0,K,1), np.mean(k_arr_single, axis=0), color = "blue", label="simulations (single source)" )
# plt.vlines(k_pred,0,1, label="Mean Passage Time", color="gray", linestyle="--")
plt.plot(np.arange(0,min(K, len(p_percolation_preds)),1)+1, p_percolation_preds[:min(K, len(p_percolation_preds))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")
# plt.plot(np.arange(0,min(K, len(p_percolation_preds)),1), np.array(p_percolation_preds[:min(K, len(p_percolation_preds))])**num_robot, color="orange", linestyle="--", label = "First Hitting Model (Multi-Source)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability ({num_robot} agents)")


plt.legend()



plt.figure(dpi=800)
plt.plot(np.arange(0,K,1), np.mean(informed_percentage_single_source, axis=0), color="blue",label="simulations (single source)" )
plt.plot(np.arange(0,K,1), np.mean(informed_percentage, axis=0), color="red", label="simulations (multi source)" )
plt.xlabel("K (time step)")
plt.ylabel("percolation percentage")
plt.title("single source percolation ("+ str(num_robot)+" agents)")

x0 = 1/num_robot
x=[x0]
for k in range(K):
    dx=0
    xk = x[-1]
    for pi in p:
        dx+= (1-(1-pi)**(xk*num_robot))*pi
    dx*=(1-xk)
        
    x.append((xk+dx))
    
plt.plot(x,"--", color = "blue", label="Mean-field-approx.")
# plt.plot((num_robot**(num_robot-2))*(1-(1-S2)**range(K))**(num_robot-1))

x0 = 1/num_robot
x=[x0]
for k in range(K):
    dx=0
    xk = x[-1]
    pk = np.linalg.matrix_power(P,k)@p0
    for pi in pk:
        dx+= (1-(1-pi)**(xk*num_robot))*pi
    dx*=(1-xk)
        
    x.append((xk+dx))
# plt.boxplot(informed_percentage_single_source[:,int(np.ceil(k_pred))], positions=[int(np.ceil(k_pred))], label="box plot")
# plt.boxplot(informed_percentage[:,int(np.ceil(k_pred))], positions=[int(np.ceil(k_pred))], label="box plot")
# plt.boxplot(informed_percentage_single_source)
plt.boxplot(informed_percentage)
# plt.plot(x,"--", color = "green", label="Mean-field-approx. (markov)")
plt.vlines(int(np.ceil(k_pred)),0,1, label="Mean Passage Time", color="gray", linestyle="--")

plt.legend()

p_first_informed = np.diff(np.mean(informed_percentage_single_source,axis=0))
p_informed_conditional = [p_first_informed[k]/(1-sum(p_first_informed[:k])) for k in range(K-1)]
plt.figure()
plt.plot(p_informed_conditional)
