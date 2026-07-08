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
import colorsys as cs

with open("percolation_trials.pkl", "rb") as file:
    graph_data = pickle.load(file)

num_method = 10
colors=[cs.hsv_to_rgb(i/num_method, 0.9, 0.9) for i in range(num_method)]

def Mk(k, pk, method, num_robot):
    M = np.zeros((num_robot,num_robot))
    p = pk[k]
    p_not_meet_conditional = (1-(np.inner(p,p)))
    for i in range(num_robot):
        for j in range(num_robot):
            I = j+1 #informed robot
            n = num_robot - I # number of uninformed robot
            if n==0:
                if i>=j:
                    M[i,j] = 1
            else:
                match method:
                    case 0:
                        p_informed_condition = kemeny_factor*(1-(p_not_meet_conditional)**I)
                    case 1:
                        p_informed_condition = (kemeny_factor_k[k])*(1-(p_not_meet_conditional)**I)
                    case 2:
                        p_informed_condition = (kemeny_factor)*np.tanh(I*kemeny_factor)*(1-(p_not_meet_conditional)**I)
                    case 3:
                        p_informed_condition = (kemeny_factor_k[k])*np.tanh(I*kemeny_factor_k[k])*(1-(p_not_meet_conditional)**I)
                    case 4:
                        p_informed_condition = (1-(p_not_meet_conditional)**(I*(kemeny_factor)))
                    case 5:
                        p_informed_condition = (1-(p_not_meet_conditional)**(I*(kemeny_factor_k[k])))
                    case 6:
                        p_informed_condition = (1-(p_not_meet_conditional)**(I*(kemeny_factor*np.tanh(I*(kemeny_factor)))))
                    case 7: 
                        p_informed_condition = (1-(p_not_meet_conditional)**(I*(kemeny_factor_k[k]*np.tanh(I*kemeny_factor_k[k]))))
                    # elif method == 6: 
                    #     p_informed_condition = (1-(p_not_meet_conditional)**(I*((num_regions/kemeny)*np.tanh(num_robot/I*num_regions/kemeny))))
                    case 8:
                        if k==0:
                            p_informed_condition = np.inner(pk[0],pk[0])
                        else:
                            p_not_met_k = (1-lambda2**k)*p_not_met[k]**(I)+lambda2**k*p_not_met[k]
                            p_not_met_k1 = (1-lambda2**(k-1))*p_not_met[k-1]**(I)+lambda2**(k-1)*p_not_met[k-1]
                            p_met_k = 1-p_not_met_k
                            p_met_k1 = 1-p_not_met_k1
                            p_first_meet_k = p_met_k - p_met_k1
                            p_informed_condition = p_first_meet_k/(p_not_met_k1)
                        
                    case 9: 
                        if k==0:
                            p_informed_condition = np.inner(pk[0],pk[0])
                        else:
                            p_not_met_k = p_not_met[k]**(I+(1-I)*a[k])
                            p_not_met_k1 = p_not_met[k-1]**(I+(1-I)*a[k-1])
                            
                            p_met_k = 1-p_not_met_k
                            p_met_k1 = 1-p_not_met_k1
                            p_first_meet_k = p_met_k - p_met_k1
                            p_informed_condition = p_first_meet_k/(p_not_met_k1)
                    
                if i>=j:
                    r = i - j  #increment of informed robot 
                    M[i,j] = math.comb(n,r)*(p_informed_condition)**r*(1-p_informed_condition)**(n-r)
    return M

def p_absorb_markov(P, p0,num_robot,  method=0):
    K0 = 999
    M0 = np.zeros((K0,num_robot,num_robot))
    
        
    for k in tqdm(range(K0)):
        M0[k,:,:] = Mk(k, pk, method, num_robot)
                
    v = np.zeros(num_robot-1)
    v[0]=1
    p_percolation_preds=[]
    prod = np.eye(num_robot-1)
    for T in np.arange(0,K0,1):
        prod = M0[T,:num_robot-1, :num_robot-1].copy()@prod
        p_percolation_preds.append(1- sum(prod@v))    
    k_pred =sum([(p_percolation_preds[k]-p_percolation_preds[k-1])*k for k in np.arange(1,K0)])
    M = M0
    return p_percolation_preds, k_pred, M


k_preds_singles = [[] for _ in range(len(graph_data))]
k_preds_all = [[] for _ in range(len(graph_data))]
p_percolation_preds_singles=[[] for _ in range(len(graph_data))]
p_percolation_preds=[[] for _ in range(len(graph_data))]
kemeny_factors = []

for i, dat in enumerate(graph_data):
    P = dat["P"]
    p0 = dat["p0"]
    num_regions = len(P)
    num_robot = dat["num_robots"]

    lambda_, _ = np.linalg.eig(P)
    idx = np.argsort(np.abs(lambda_))
    lambda2 = np.abs(lambda_[idx[-2]])
    a = [lambda2**k for k in range(9999)]

    P_infty = np.linalg.matrix_power(P, 9999)
    p_infty = P_infty[:,0]
    P_perp = P-P_infty
    P_perp_k = [np.linalg.matrix_power(P_perp,k) for k in range(1000)]
    Z_k = [np.sum(P_perp_k[:k+1], axis=0) for k in range(999)]
    Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
    kemeny = np.trace(Z)
    kemeny_k = np.array([np.trace(x) for x in Z_k])
    kemeny_factor = num_regions/kemeny
    kemeny_factor_k = num_regions/kemeny_k
    kemeny_factors.append(kemeny_factor)
    pk=[p0.copy()]
    
    q_state = []
    idx = []
    for ii in range(num_regions):
        for j in range(num_regions):
            if ii==j:
                idx.append(num_regions*ii+j)
            else:    
                q_state.append((ii,j))

 
    p0_kron = np.kron(p0, p0)
    # p_infty_kron =np.kron(p_infty,p_infty) 
    P_kron= np.kron(P,P)

    Q = P_kron.copy()
    Q=np.delete(Q,idx, 0)
    Q=np.delete(Q,idx, 1)

    q0 = p0_kron.copy()
    q0 = np.delete(q0,idx)

    # p_kron = [p0_kron]
    # for k in range(1000):
    #     p_kron.append(P_kron@p_kron[-1].copy())
        
    q = [q0]
    for k in range(1000):
        q.append(Q@q[-1].copy())
        


        

    p_not_met = np.sum(q, axis=1)
    
    for k in range(1000):
        pk.append(P@pk[k].copy())
        
    for method in range(num_method):
        p_percolation_preds_single, k_pred, M = p_absorb_markov(P.copy(), p0.copy(), num_robot, method)
        k_preds_singles[i].append(k_pred)
        p_percolation_preds_singles[i].append(p_percolation_preds_single)

        percolation_percent_expected =[1/num_robot]
        v=np.zeros(num_robot)
        v[0]=1
        for k in range(999):
            v=M[k,:,:]@v
            percolation_percent_expected.append(1/num_robot*np.inner(v,np.arange(1,num_robot+1)))
        
        p_percolation_preds[i].append(np.array([p**(num_robot-1) for p in p_percolation_preds_single]))
    
    K_plot = int(min(3*max(k_preds_singles[i]),len(M)))
        
        
    plt.figure(dpi=800)
    plt.plot(np.arange(0,K_plot,1), dat["p percolation single"][0:K_plot], color = "k", label="simulations" )
    for ii, p_percolation_preds_single in enumerate(p_percolation_preds_singles[i]):
        plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds_single)),1), p_percolation_preds_single[:min(K_plot, len(p_percolation_preds_single))], linestyle="--", label = f"method{ii}", color = colors[ii])
    
    plt.xlabel("k (time step)")
    plt.ylabel("Full Percolation Probability ")
    plt.title(f"Percolation Probability (single source, {num_robot} agents, {num_regions} regions)")
    plt.legend()
    
    
    plt.figure(dpi=800)
    plt.plot(np.arange(0,K_plot,1), dat["p percolation all"][0:K_plot], color = "k", label="simulations" )
    for ii, p_percolation_pred in enumerate(p_percolation_preds[i]):
        plt.plot(np.arange(0,min(K_plot, len(p_percolation_pred)),1), p_percolation_pred[:min(K_plot, len(p_percolation_pred))], linestyle="--", label = f"method{ii}", color = colors[ii])
    
    plt.xlabel("k (time step)")
    plt.ylabel("Full Percolation Probability ")
    plt.title(f"Percolation Probability (all source, {num_robot} agents, {num_regions} regions)")
    plt.legend()
    
    plt.show()
    
data = {"p percolation single": p_percolation_preds_singles,
        "p percolation all": p_percolation_preds}
with open("percolation_trials_pred.pkl", "wb") as file:
    pickle.dump(data, file)
    
#%%

with open("percolation_trials_pred.pkl", "rb") as file:
    data = pickle.load(file)
    
graph_types = ["er-directed", "er-undirected", "cycle", "barbell", "grid"]
color_type = [cs.hsv_to_rgb(i/num_method, 0.9, 0.9) for i in range(len(graph_types))]

p_percolation_preds_singles = data["p percolation single"]    
p_percolation_preds = data["p percolation all"]  
kemeny_factors = []
lambda2s = []
num_robots = []
graph_type = []
for i, dat in enumerate(graph_data):
    P = dat["P"]
    
    num_regions = len(P)
    num_robots.append(dat["num_robots"])
    graph_type.append(dat["type"])
    lambda_, _ = np.linalg.eig(P)
    idx = np.argsort(np.abs(lambda_))
    lambda2s.append(np.abs(lambda_[idx[-2]]))
    P_infty = np.linalg.matrix_power(P, 9999)
    P_perp = P-P_infty
  
    Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
    kemeny = np.trace(Z)
    kemeny_factors.append(num_regions/kemeny)

kemeny_factors = np.array(kemeny_factors)
lambda2s =  np.array(lambda2s)
num_robots = np.array(num_robots)
graph_type = np.array(graph_type)

for i, type_ in enumerate(graph_types):
    for j, entry in enumerate(graph_type):
        if entry == type_:
            graph_type[j]= int(i)
graph_type = graph_type.astype(np.int32)
k_pred_all = np.zeros((len(graph_data), num_method))
for i in range(len(graph_data)):
    for j in range(num_method):
        p = p_percolation_preds[i][j]**2
        k_pred_all[i,j] = sum([(p[k]-p[k-1])*k for k in np.arange(1,len(p))])

# k_preds=np.array(k_preds_singles)

k_preds = np.zeros((len(graph_data), num_method))
for i in range(len(graph_data)):
    for j in range(num_method):
        p = np.array(p_percolation_preds_singles[i][j])**2
        k_preds[i,j] = sum([(p[k]-p[k-1])*k for k in np.arange(1,len(p))])
        
kc_single_sim = np.array([dat["kc single"] for dat in graph_data])
kc_all_sim = np.array([dat["kc all"] for dat in graph_data])


plt.figure()
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# for method in range(num_method):
#     plt.plot(lambda2s, (k_preds[:,method]-kc_single_sim)/kc_single_sim, ".", color=colors[method], label=f"{method}")
plt.plot(kemeny_factors, (k_preds[:,5]-kc_single_sim)/kc_single_sim, ".", color=colors[5], label=f"{5}")
plt.plot(kemeny_factors, (k_preds[:,9]-kc_single_sim)/kc_single_sim, ".", color=colors[9], label=f"{9}")


plt.xlabel(r"$\lambda_2$")

plt.ylabel(r"$(K_{c,pred}-K_c)/K_c$")
plt.legend()
plt.title("single source")


plt.figure()
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# for method in range(num_method):
#     plt.plot(kemeny_factors, (k_pred_all[:,method]-kc_all_sim)/kc_all_sim, ".", color=colors[method], label=f"{method}")

plt.plot(lambda2s, (k_pred_all[:,1]-kc_all_sim)/kc_all_sim, ".", color=colors[5], label=f"{1}")
plt.plot(lambda2s, (k_pred_all[:,9]-kc_all_sim)/kc_all_sim, ".", color=colors[9], label=f"{9}")


plt.xlabel("Normalized Kemeny Factor")
plt.ylabel(r"$(K_{c,pred}-K_c)/K_c$")
plt.legend()
plt.title("all source")

plt.figure()
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
method = 9
for i in  range(len(graph_types)):
    idx = np.where(np.array(graph_type)==i)[0]
    plt.plot(lambda2s[idx], (k_preds[idx,method]-kc_single_sim[idx])/kc_single_sim[idx], ".", color=color_type[i], label=graph_types[i])

plt.xlabel(r"$\lambda_2$")

plt.ylabel(r"$(K_{c,pred}-K_c)/K_c$")
plt.legend()
plt.title("single source")
