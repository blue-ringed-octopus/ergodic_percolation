# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:17:30 2025

@author: hibado
"""


import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt 
import pickle 
num_robot = 5
num_trials = 100
num_p = 30
num_regions = 100
K = 50
critical=np.zeros((num_p,num_trials))*np.nan

p_arr = []
for i in range(num_p):
    p = np.random.rand(num_regions)
    p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
    p_arr.append(p)


for ii, p in enumerate(p_arr):
    for trial in range(num_trials):
        G=nx.DiGraph()
#        reachable = np.zeros(num_robot)
#        reachable[0] = 1
        reachable = np.eye(num_robot)
        for k in range(K):
            r = np.random.choice(range(num_regions), num_robot, p=p)
            for i in range(num_robot):
                G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k+1)+"]")
                G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
    
            for i,j in combinations(range(num_robot), 2): 
                if r[i] == r[j]:
                    G.add_edge(str(i)+"["+str(k)+"]", str(j)+"["+str(k)+"]")
                    G.add_edge(str(j)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
            
            # for i in range(num_robot):
            #    for j in range(num_robot):
            #        if not reachable[i,j]:
            #            reachable[i,j] = nx.has_path(G,str(i)+"["+str(0)+"]", str(j)+"["+str(k)+"]")

            # if reachable.all():
            #     critical[ii,trial]=k+1
            #     break
            for i in [0]:
                for j in range(num_robot):
                    if not reachable[i,j]:
                        reachable[i,j] = nx.has_path(G,str(0)+"["+str(0)+"]", str(j)+"["+str(k)+"]")
            if reachable[i,:].all():
                critical[ii, trial]=k+1
                break
#%%
ptps = [np.inner(p,p) for p in p_arr]
p = np.linspace(0,1,100)

def p_connected(ptp, N, K):
    if np.isnan(p_con[N,K]):
        if K==0:
            return 0
        if K==1:
            return ptp
        if N ==2 :
            return 1-(1-ptp)**K
        p = 1
        for n in np.arange(2,N-2,1):
            for k in  np.arange(2,K,1):
                p*= (1-ptp*p_connected(ptp, n,k))
        p_con[N,K] = 1 - (1-ptp)**K*p
    return p_con[N,K].copy()

def k_critical(ptp):
    global p_con
    p_con=np.zeros((num_robot+100,1000))*np.nan

    thres = 0.5
    if ptp==0:
        return np.inf
    if ptp>=thres:
        return 1
    ps=[ptp]
    k=1
    while ps[-1]<thres:
        k+=1
        p = p_connected(ptp, num_robot, k)
        ps.append(p)

    return (k-1)+1/(ps[-1]-ps[-2])*(thres-ps[-2])

k_c = np.mean(critical, axis=1)
k_c_pred = [k_critical(ptp) for ptp in p]
plt.figure()
plt.plot(ptps, k_c, '.', color="r", alpha=1)
plt.plot(p, k_c_pred, '--', color="k")
plt.vlines(0,np.nanmax(critical), 1/num_regions, linestyle='--')
plt.xlabel(r"$p^Tp$")
plt.ylabel(r"$K_c$")
plt.yticks(np.arange(0,int(np.nanmax(critical)),3))
plt.title("Kc vs pTp (n="+str(num_robot)+")")
plt.legend()

#%%

p_con=np.zeros((num_robot+1,100))*np.nan
trial=6
K = 20
k_mean = np.zeros(K)
k_arr = []
for k in critical[trial]:
    ki = np.zeros(K)
    ki[int(k):] = 1
    k_arr.append(ki)
    k_mean += ki/len(critical[trial])
p_pred = [p_connected(ptps[trial], num_robot, k) for k in range(K)]

plt.figure()
plt.plot(range(K),k_mean, ".")
plt.plot(range(K),p_pred, "--")
plt.title("pTp="+str(ptps[trial]))
# plt.vlines(kc,0,1, linestyle="--", color="r")
# plt.vlines(k_c[trial],0,1, linestyle="--", color="b")
#%%
# k_arr = np.array(k_arr)
# mu = np.nanmean(critical[trial])
# k_shift = critical[trial] - mu
# k_shift = k_shift[~np.isnan(k_shift)]
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression() 
# model.fit((np.outer(np.ones(len(k_arr)), range(K))).reshape(-1,1), k_arr.reshape(-1))
# y_pred = model.predict_proba(np.array(range(K)).reshape(-1,1))[:, 1]
# plt.figure()
# plt.plot(range(K),k_mean, ".")
# plt.plot(range(K),y_pred, "--")
# plt.title("pTp="+str(ptps[trial]))
# dat = {"p":p, "K": critical,"N": num_robots }
# with open('data/3.p', 'wb') as file:
#      pickle.dump(dat, file)