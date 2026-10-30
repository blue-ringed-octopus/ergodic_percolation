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
import os
from absorbing_model import Absoorbing_Model

output_name = "resources/percolation_trials_pred_complete.pkl"
with open("resources/percolation_trials_complete.pkl", "rb") as file:
    graph_data = pickle.load(file)
    
    
if os.path.isfile(output_name):    
    with open(output_name, "rb") as file:
        data = pickle.load(file)
else:
    data = {"p percolation single": [],
            "p percolation all": []}
    
num_method = 1

colors=[cs.hsv_to_rgb(i/num_method, 0.9, 0.9) for i in range(num_method)]


p_percolation_preds_singles = data["p percolation single"] 
p_percolation_preds = data["p percolation all"]

p_percolation_preds_singles+=[[] for _ in range(len(graph_data)- len(p_percolation_preds_singles))]
p_percolation_preds+=[[] for _ in range(len(graph_data)- len(p_percolation_preds))]

kemeny_factors = []

for i, dat in enumerate(tqdm(graph_data)):
    if True:#len(p_percolation_preds_singles[i]) < num_method:
        P = dat["P"]
        p0 = dat["p0"]
        num_robot = dat["num_robots"]
        k_preds_singles=[]
        model = Absoorbing_Model(P, p0, num_robot)
        for method in np.arange(len(p_percolation_preds_singles[i]),num_method):
            p_percolation_preds_single, k_pred, M = model.pred(method)
            k_preds_singles.append(k_pred)
            p_percolation_preds_singles[i].append(p_percolation_preds_single)
    
            percolation_percent_expected =[1/num_robot]
            v=np.zeros(num_robot)
            v[0]=1
            for Mk in M:
                v=Mk@v
                percolation_percent_expected.append(1/num_robot*np.inner(v,np.arange(1,num_robot+1)))
            
            p_percolation_preds[i].append(np.array([p**(num_robot-1) for p in p_percolation_preds_single]))
        
    
data = {"p percolation single": p_percolation_preds_singles,
        "p percolation all": p_percolation_preds}
with open(output_name, "wb") as file:
    pickle.dump(data, file)
    
#%%

# for i, dat in enumerate(graph_data):
#     plt.figure(dpi=800)
#     plt.plot(np.arange(0,K_plot,1), dat["p percolation single"][0:K_plot], color = "k", label="simulations" )
#     for ii, p_percolation_preds_single in enumerate(p_percolation_preds_singles[i]):
#         plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds_single)),1), p_percolation_preds_single[:min(K_plot, len(p_percolation_preds_single))], linestyle="--", label = f"method{ii}", color = colors[ii])
    
#     plt.xlabel("k (time step)")
#     plt.ylabel("Full Percolation Probability ")
#     plt.title(f"Percolation Probability (single source, {num_robot} agents, {num_regions} regions)")
#     plt.legend()
    
with open(output_name, "rb") as file:
    data = pickle.load(file)


graph_types = ["complete"]
# color_type = [cs.hsv_to_rgb(i/num_method, 0.9, 0.9) for i in range(len(graph_types))]

p_percolation_preds_singles = data["p percolation single"]    
p_percolation_preds = data["p percolation all"]  
num_method = len(p_percolation_preds_singles[0])
kemeny_factors = []
lambda2s = []
num_robots = []
graph_type = []
clustering_factor= []

for i, dat in enumerate(graph_data):
    P = dat["P"]
    num_regions = len(P)
    num_robots.append(dat["num_robots"])
    graph_type.append(dat["type"])
    lambda_, v = np.linalg.eig(P)
    idx = np.argsort(np.abs(lambda_))
    v1 = v[idx[-1]]
    v1 = v1/sum(v1)
    lambda2s.append(np.abs(lambda_[idx[-2]]))
    P_infty = np.linalg.matrix_power(P, 9999)
    rho_bar=P_infty[:,0]
    clustering_factor.append(num_robots[i]*np.inner(rho_bar,rho_bar))

    P_perp = P-P_infty
    Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
    kemeny = np.trace(Z)
    kemeny_factors.append(num_regions/kemeny)

kemeny_factors = np.array(kemeny_factors)
lambda2s =  np.array(lambda2s)
num_robots = np.array(num_robots)
clustering_factor = np.array(clustering_factor)
# graph_type = np.array(graph_type)

for i, type_ in enumerate(graph_types):
    for j, entry in enumerate(graph_type):
        if entry == type_:
            graph_type[j]= int(i)
# graph_type = graph_type.astype(np.int32)
k_pred_all = np.zeros((len(graph_data), num_method))
for i in range(len(graph_data)):
    for j in range(num_method):
        p = p_percolation_preds[i][j]
        k_pred_all[i,j] = sum([(p[k]-p[k-1])*k for k in np.arange(1,len(p))])

# k_preds=np.array(k_preds_singles)

k_preds = np.zeros((len(graph_data), num_method))
for i in range(len(graph_data)):
    for j in range(num_method):
        p = np.array(p_percolation_preds_singles[i][j])
        k_preds[i,j] = sum([(p[k]-p[k-1])*k for k in np.arange(1,len(p))])
        
kc_single_sim = np.array([dat["kc single"] if dat["p percolation single"][-1]>0.9 else np.nan for dat in graph_data])
kc_all_sim = np.array([dat["kc all"] if dat["p percolation all"][-1]>0.9 else np.nan for dat in graph_data])
density = num_robots/[len(dat["P"]) for dat in graph_data]
# rmse = [[np.sqrt(np.mean((dat["p percolation single"][:999] - p_percolation_preds_singles[i][j]) ** 2)) for i,dat in enumerate(graph_data)] for j in range(num_method)]
#%%
fontsize = 12
error_all = (k_pred_all[:,0]-kc_all_sim)/kc_all_sim
error_single = (k_preds[:,0]-kc_single_sim)/kc_single_sim

plt.figure(dpi=800)
plt.hlines(0,0.5,2.5, linestyle="--", color="grey")
plt.boxplot([error_single,error_all] , tick_labels=["single source", "all source"])
plt.title(r"$\bf{k_c}$ Relative Error"+f" ({len(k_preds)} Trials)", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(k_{c, pred}-k_c)/k_c}$", weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

method = 0
# idx = graph_type==4
plt.figure(dpi=800)
# plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
plt.plot(clustering_factor, kc_single_sim, ".", label=f"", color="black", alpha=0.5, mew=0)
plt.plot(clustering_factor, k_preds[:,method], ".", label=f"", color="red", alpha=0.5, mew=0)

plt.xlabel(r"Clustering Factor $\bf{(NS_2)}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")
# plt.ylim([0,100])


plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=max(clustering_factor),linestyle="--", color="grey")
plt.plot(clustering_factor, (k_preds[:,method]-kc_single_sim)/kc_single_sim, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
plt.xlabel(r"Clustering Factor $\bf{(NS_2)}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Percolation Time Prediction Error \n (Single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")
# plt.ylim([-1,1])
plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=max(clustering_factor),linestyle="--", color="grey")

plt.plot(clustering_factor, (k_pred_all[:,method]-kc_all_sim)/kc_all_sim, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
plt.xlabel(r"Clustering Factor $\bf{(NS_2)}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Percolation Time Prediction Error \n (All Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

# err_1 = np.abs((k_preds[:,method2]-kc_single_sim)/kc_single_sim)

# plt.figure()
# plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# plt.plot(lambda2s, err_1-err_baseline, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
# plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
# plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
# plt.legend(fontsize=fontsize,prop={'weight':"bold"})
# plt.title("Pairwise \n (single Source)", fontsize=fontsize, weight="bold")
# plt.xticks(fontsize=fontsize, weight="bold")
# plt.yticks(fontsize=fontsize, weight="bold")
# plt.ylim([-1,1])
# plt.figure()
# plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# plt.plot(lambda2s, (k_pred_all[:,0]-kc_all_sim)/kc_all_sim, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
# plt.plot(lambda2s, (k_pred_all[:,method2]-kc_all_sim)/kc_all_sim, ".", color=colors[method2], label=r"$\lambda_2-$Corrected", alpha=0.5, mew=0)
# plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
# plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
# plt.legend(fontsize=fontsize,prop={'weight':"bold"})
# plt.title("Relative Percolation Time Prediction Error \n (All Source)", fontsize=fontsize, weight="bold")
# plt.xticks(fontsize=fontsize, weight="bold")
# plt.yticks(fontsize=fontsize, weight="bold")

# err_baseline = np.abs((k_pred_all[:,0]-kc_all_sim)/kc_all_sim)
# err_1 = np.abs((k_pred_all[:,method2]-kc_all_sim)/kc_all_sim)
# plt.figure()
# plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# plt.plot(lambda2s, err_1-err_baseline, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
# plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
# plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
# plt.legend(fontsize=fontsize,prop={'weight':"bold"})
# plt.title("Pairwise \n (All Source)", fontsize=fontsize, weight="bold")
# plt.xticks(fontsize=fontsize, weight="bold")
# plt.yticks(fontsize=fontsize, weight="bold")
