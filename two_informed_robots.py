# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:52:58 2026

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
num_robot = 3
num_trial = 9999
I = num_robot-1

 

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
        
g = nx.grid_graph((4,4), periodic = True)
# g = nx.wheel_graph(num_regions)
# g = nx.hexagonal_lattice_graph(5,6, periodic = True)
# g = nx.cycle_graph(num_regions)
# g = nx.barbell_graph(5,3)
num_regions = len(g)
A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
# A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,3)], axis=0).astype(bool)

P = np.zeros((num_regions, num_regions))
for i in range(num_regions):
   idx =  np.where(A[:,i])[0]
   P[idx, i] = dirichlet.rvs(np.ones(len(idx)))[0]

# p_infty = np.linalg.matrix_power(P,9999)[:,0]
# p0=p_infty
p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
#%%
lambda_, v = np.linalg.eig(P)
idx = np.argsort(np.abs(lambda_))
lambda2 = np.abs(lambda_[idx[-2]])

P_infty = np.linalg.matrix_power(P, 9999)
p_infty =P_infty[:,0].copy()
P_perp = P-P_infty
# P_perp_k = [np.linalg.matrix_power(P_perp,k) for k in range(2000)]
# Z_k = [np.sum(P_perp_k[:k+1], axis=0) for k in range(1999)]
Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
kemeny = np.trace(Z)
# kemeny_k = np.array([np.trace(x) for x in Z_k])
# kemeny_k = np.array([np.max(np.diag(x)) for x in Z_k])*num_regions
# kemeny_k = np.array([np.quantile(np.diag(x), 0.9) for x in Z_k])*num_regions
# kemeny_k = np.array([np.mean(np.diag(x))+1*np.std(np.diag(x)) for x in Z_k])*num_regions
# eig_perp = np.linalg.eigvals(Z)
# test = np.array([np.linalg.norm(x) for x in Z_k])
# plt.plot(num_regions/kemeny_k)


# test = lambda_
# for i in tqdm(np.arange(0,num_robot-1)):
#     test = np.kron(test,lambda_)
    
# test[1-np.abs(test)<0.0001] = 0
# kemeny_k_test = [sum([(1-l**k)/(1-l) for l in test]) for k in np.arange(1,10)]
p_percolation_preds = np.ones(1000)

pk=[p0.copy()]
for k in range(1000):
    pk.append(P@pk[k].copy())

pk = np.array(pk)
S2 = np.array([sum(p**2) for p in pk])
#%% simulation
p_I = np.zeros(num_regions)
p_I[0] = 1

K = 100
rho = np.zeros((num_trial, K, num_regions))
kc = np.ones(num_trial)*np.nan

rho_I = np.zeros((num_trial, K, num_regions)) 

for trial in tqdm(range(num_trial)):
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    r[1:] =  np.random.choice(range(num_regions), num_robot-1, p=p_I)
    # # r[1]=0 np.random.choice(range(num_regions), num_robot, p=p0)
    # for i in np.arange(2,num_robot):
    #     r[i]=r[1]
        
        
    for k in range(K): 
        rho_I[trial,k,r[1:]] += 1
            
        if sum([r[0]==r[i] for i in np.arange(1,num_robot)]):
            kc[trial] = min(k, kc[trial])
            break
            
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]    

            rho[trial,k,r[robot]]+=1/num_robot

rho = np.mean(rho,axis=0)

# rho_I_bar = np.mean(rho_I,axis=0)/I
# err = np.mean(np.linalg.norm(rho_I/I-pk[:100], axis=2),axis=0)
# plt.figure()
# plt.plot(err)
# plt.plot([err[-1]+(err[0]-err[-1])*(0.8)**k for k in range(K)])
# plt.title("Average error of distribution of informed robots")  


p_first_meet_sim = np.array([sum(kc==i) for i in range(K)])/num_trial
p_met_sim = np.cumsum(p_first_meet_sim)
p_meet_conditional_sim = [p_first_meet_sim[k] if k==0 else p_first_meet_sim[k]/(1-p_met_sim[k-1]) for k in range(K)]
#%% firt meeting model
tau = 1/(1-lambda2)
# tau = -1/np.log(lambda2)
a = 1-num_regions/kemeny
a = [lambda2**k for k in range(9999)]


# a = []
# for i in range(num_regions):
#     test = np.zeros(num_regions)
#     test[i] = 1
#     a.append(np.linalg.norm(P_perp@test,1)/np.linalg.norm(test-p_infty,1))
# a = np.inner(p0,a)

# a = [np.linalg.norm(np.linalg.matrix_power(P_perp,k)@(p_I-p_infty),1)/np.linalg.norm(p_I-p_infty,1) for k in np.arange(0,1000)]

# a = [np.linalg.norm(np.linalg.matrix_power(P,k)@(p_I)-(p_infty),1)/np.linalg.norm(p_I-p_infty,1) for k in np.arange(0,1000)]

q_state = []
idx = []
for i in range(num_regions):
    for j in range(num_regions):
        if i==j:
            idx.append(num_regions*i+j)
        else:    
            q_state.append((i,j))

# idx = [num_regions*i+i for i in range(num_regions)]

# p0_kron = np.kron(p0, p0)
p0_kron = np.kron(p_I, p0)
p_infty_kron =np.kron(p_infty,p_infty) 
P_kron= np.kron(P,P)



Q = P_kron.copy()
Q=np.delete(Q,idx, 0)
Q=np.delete(Q,idx, 1)

q0 = p0_kron.copy()
q0 = np.delete(q0,idx)

Q_2 = np.kron(P,Q)
q0_2 = np.kron(p_I, q0)

q_state2 = []
idx2 = []
n=0
for i in range(num_regions):
    for j,k in q_state:
        if i==k:
            idx2.append(n)
        else:    
            q_state2.append((i,j,k))
        n+=1

p_kron = [p0_kron]
for k in range(1000):
    p_kron.append(P_kron@p_kron[-1].copy())
    
q = [q0]
for k in range(1000):
    q.append(Q@q[-1].copy())
    
# idx2 =  [len(q0)*i+i for i in range(len(p0))]

Q_2 = np.delete(Q_2, idx2,0)
Q_2 = np.delete(Q_2, idx2,1)
q0_2 = np.delete(q0_2, idx2)

q2 = [q0_2]
for k in range(1000):
    q2.append(Q_2@q2[-1].copy())
# a = [np.linalg.norm(p_kron[k]-(p_infty_kron),1)/np.linalg.norm(p0_kron-p_infty_kron,1) for k in np.arange(0,1000)]


    

p_not_met = np.sum(q, axis=1)
p_not_met_n = p_not_met**(I)
p_not_met_3 = np.sum(q2, axis=1)
# p_not_met_corrected = np.array([p_not_met[k]**(I+(1-I)*np.exp(-k/(tau))) for k in range(K)])
p_not_met_corrected = np.array([p_not_met[k]**(I+(1-I)*a[k]) for k in range(K)])
p_not_met_interp =  np.array([(1-a[k])*p_not_met_n[k]+ a[k]*p_not_met[k] for k in range(K)])


p_met = 1-p_not_met
p_met_n = 1-p_not_met_n
p_met_3 = 1-p_not_met_3
p_met_corrected = 1 - p_not_met_corrected
p_met_interp = 1- p_not_met_interp


p_first_meet = np.concatenate(([p_met[0]], np.diff(p_met)))
p_first_meet_n = np.concatenate(([p_met_n[0]], np.diff(p_met_n)))
p_first_meet_3 = np.concatenate(([p_met_3[0]], np.diff(p_met_3)))
p_first_meet_corrected = np.concatenate(([p_met_corrected[0]], np.diff(p_met_corrected)))
p_first_meet_interp = np.concatenate(([p_met_interp[0]], np.diff(p_met_interp)))

p_meet_conditional = [p_first_meet[k] if k==0 else p_first_meet[k]/(p_not_met[k-1]) for k in range(K)]
p_meet_conditional_n = np.array([p_first_meet_n[k] if k==0 else p_first_meet_n[k]/(p_not_met_n[k-1]) for k in range(K)])
p_meet_conditional_3 = np.array([p_first_meet_3[k] if k==0 else p_first_meet_3[k]/(p_not_met_3[k-1]) for k in range(K)])

p_meet_conditional_corrected = np.array([p_first_meet_corrected[k] if k==0 else p_first_meet_corrected[k]/(p_not_met_corrected[k-1]) for k in range(K)])
p_meet_conditional_interp = np.array([p_first_meet_interp[k] if k==0 else p_first_meet_interp[k]/(1-p_met_interp[k-1]) for k in range(K)])

#%%
# plt.figure()
# plt.plot(S2[:K])
# plt.plot(P_meet[:K])
# plt.title("meeting chance")
K_plot=10
plt.figure()
plt.plot(p_meet_conditional[:K_plot], "b", label="(1 informed)")
plt.plot(p_meet_conditional_n[:K_plot], "g",label="(n independent)")
plt.plot(p_meet_conditional_3[:K_plot], label="(3)")
plt.plot(p_meet_conditional_corrected[:K_plot], "r",label="pred (corrected)")
plt.plot(p_meet_conditional_interp[:K_plot], "orange",label="pred (interp)")
plt.plot(1-(1-S2[:K_plot])**I, label = "unconditioned meeting")
plt.plot(p_meet_conditional_sim[:K_plot],"k--", label="sim")
plt.title("P(meet) (conditional)")
plt.legend()

plt.figure()
plt.plot(p_first_meet[:K_plot], "b", label="first meet model (1 informed)" )
plt.plot(p_first_meet_n[:K_plot], "g", label="first meet model (n independent)" )
plt.plot(p_first_meet_3[:K_plot],  label="first meet model (3)" )
plt.plot(p_first_meet_corrected[:K_plot], "r", label="first meet model (corrected)" )
plt.plot(p_first_meet_interp[:K_plot], "orange", label="first meet model (interp)" )

plt.plot(p_first_meet_sim[:K_plot],"k--", label="sim")
plt.legend()
plt.title("first meet")

plt.figure()
plt.plot(p_met[:K_plot],"b", label="first meet model (1 informed)")
plt.plot(p_met_n[:K_plot],"g", label="first meet model (n independent)")
plt.plot(p_met_3[:K_plot], label="first meet model (3 independent)")
plt.plot(p_met_corrected[:K_plot],"r", label="first meet model corrected")
plt.plot(p_met_interp[:K_plot],"orange", label="interpolation")
plt.plot(p_met_sim[:K_plot],"k--", label="sim")
plt.legend()
plt.title("Met at least once")

