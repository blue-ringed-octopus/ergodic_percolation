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
from time import time 

num_method = 13

with open("resources/percolation_trials.pkl", "rb") as file:
    graph_data = pickle.load(file)
    
    
if os.path.isfile("resources/percolation_trials_pred.pkl"):    
    with open("resources/percolation_trials_pred.pkl", "rb") as file:
        data = pickle.load(file)
else:
    dat = {"p percolation single": None,
            "p percolation all": None,
            "informed percentage": None}
    
    data = {}
    for graph in range(len(graph_data)):
        data[graph] = {}
        for method in range(num_method):
            data[graph][method] = dat.copy()



# p_percolation_preds_singles = data["p percolation single"] 
# p_percolation_preds = data["p percolation all"]

# p_percolation_preds_singles+=[[] for _ in range(len(graph_data)- len(p_percolation_preds_singles))]
# p_percolation_preds+=[[] for _ in range(len(graph_data)- len(p_percolation_preds))]


for i, dat in enumerate(tqdm(graph_data)):
    P = dat["P"]
    p0 = dat["p0"]
    num_regions = len(P)
    num_robot = dat["num_robots"]
    if dat['p percolation single'][-1]>0.9:
        model = Absoorbing_Model(P, p0, num_robot)
        k_preds_singles = []
        t = time()
        for method in range(num_method):
            if method==12:
                continue 
            if data[i][method]['p percolation single'] is None:
                k_max = len(dat['p percolation single'])
                p_percolation_preds_single, k_pred, M = model.pred(method, k_max)
                k_preds_singles.append(k_pred)
                

                percolation_percent_expected =[1/num_robot]
                v=np.zeros(num_robot)
                v[0]=1
                for Mk in M:
                    v=Mk@v
                    percolation_percent_expected.append(1/num_robot*np.inner(v,np.arange(1,num_robot+1)))
                data[i][method]['p percolation single'] = p_percolation_preds_single
                data[i][method]['p percolation all'] = np.array([p**(num_robot-1) for p in p_percolation_preds_single])
                data[i][method]['informed percentage'] = percolation_percent_expected
                
                print(method, time()-t)
                t = time()
            with open("resources/percolation_trials_pred.pkl", "wb") as file:
                pickle.dump(data, file)
    


    
#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle 
import colorsys as cs
import os
from absorbing_model import Absoorbing_Model
# for i, dat in enumerate(graph_data):
#     plt.figure(dpi=800)
#     plt.plot(np.arange(0,K_plot,1), dat["p percolation single"][0:K_plot], color = "k", label="simulations" )
#     for ii, p_percolation_preds_single in enumerate(p_percolation_preds_singles[i]):
#         plt.plot(np.arange(0,min(K_plot, len(p_percolation_preds_single)),1), p_percolation_preds_single[:min(K_plot, len(p_percolation_preds_single))], linestyle="--", label = f"method{ii}", color = colors[ii])
    
#     plt.xlabel("k (time step)")
#     plt.ylabel("Full Percolation Probability ")
#     plt.title(f"Percolation Probability (single source, {num_robot} agents, {num_regions} regions)")
#     plt.legend()
    
with open("resources/percolation_trials.pkl", "rb") as file:
    graph_data = pickle.load(file)
    
with open("resources/percolation_trials_pred.pkl", "rb") as file:
    data = pickle.load(file)

method1 = 10
method2 = 1
graph_types = ["er-directed", "er-undirected", "cycle", "barbell", "grid"]

p_percolation_preds_singles = [[graph[method]["p percolation single"] for method in graph.keys()] for graph in data.values()]
p_percolation_preds_all = [[graph[method]["p percolation all"] for method in graph.keys()] for graph in data.values()]
num_method = len(data[0])
color_type = [cs.hsv_to_rgb(i/len(graph_types), 0.9, 0.9) for i in range(len(graph_types))]
colors = [cs.hsv_to_rgb(i/num_method, 0.9, 0.9) for i in range(num_method)]

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
        p = p_percolation_preds_all[i][j]
        if p is None:
            k_pred_all[i,j] = None
        else:
            k_pred_all[i,j] = np.inner(np.diff(p), np.arange(1, p.size) )

            # k_pred_all[i,j] = sum([(p[k]-p[k-1])*k for k in np.arange(1,p.size)])

# k_preds=np.array(k_preds_singles)

k_preds = np.zeros((len(graph_data), num_method))
for i in range(len(graph_data)):
    for j in range(num_method):
        p = p_percolation_preds_singles[i][j]
        if p is None:
            k_preds[i,j] = None
        else:
            k_preds[i,j] = np.inner(np.diff(p), np.arange(1, len(p)) )
        
kc_single_sim = np.array([dat["kc single"] if dat["p percolation single"][-1]>0.9 else np.nan for dat in graph_data])
kc_all_sim = np.array([dat["kc all"] if dat["p percolation all"][-1]>0.9 else np.nan for dat in graph_data])

# rmse = [[np.sqrt(np.mean((dat["p percolation single"][:999] - p_percolation_preds_singles[i][j]) ** 2)) for i,dat in enumerate(graph_data)] for j in range(num_method)]
#%%
fontsize = 12

plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
for graph in range(len(graph_types)):
    idx = graph_type==graph
    plt.plot(lambda2s[idx], (k_preds[idx,0]-kc_single_sim[idx])/kc_single_sim[idx], ".", label=graph_types[graph], alpha=1, mew=0)
plt.xlabel(r"$\bf{|\lambda_2(P)|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Error of Independent Model \n (single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
for graph in range(len(graph_types)):
    idx = graph_type==graph
    plt.plot(lambda2s[idx], (k_preds[idx,1]-kc_single_sim[idx])/kc_single_sim[idx], ".", label=graph_types[graph], alpha=1, mew=0)
plt.xlabel(r"$\bf{|\lambda_2(P)|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Error of Kemeny Factor Model \n (single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
for graph in range(len(graph_types)):
    idx = graph_type==graph
    plt.plot(lambda2s[idx], (k_preds[idx,10]-kc_single_sim[idx])/kc_single_sim[idx], ".", label=graph_types[graph], alpha=1, mew=0)
plt.xlabel(r"$\bf{|\lambda_2(P)|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Error of Dependent Model \n (single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
for graph in range(len(graph_types)):
    idx = graph_type==graph
    plt.plot(lambda2s[idx], (k_preds[idx,12]-kc_single_sim[idx])/kc_single_sim[idx], ".", label=graph_types[graph], alpha=1, mew=0)
plt.xlabel(r"$\bf{|\lambda_2(P)|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Error of Dependent Model \n (single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")




idx = graph_type==4 
plt.figure(dpi=800)
# plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
plt.plot(lambda2s[idx], kc_single_sim[idx], ".", label=f"", color="black", alpha=0.5, mew=0)
plt.plot(lambda2s[idx], k_preds[idx,method2], ".", label=f"", color="red", alpha=0.5, mew=0)

plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")
plt.ylim([0,100])


plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
# for method in range(num_method):
#     plt.plot(lambda2s, (k_preds[:,method]-kc_single_sim)/kc_single_sim, ".", color=colors[method], alpha=0.5, label=f"{method}", mew=0)
plt.plot(lambda2s, (k_preds[:,0]-kc_single_sim)/kc_single_sim, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
plt.plot(lambda2s, (k_preds[:,10]-kc_single_sim)/kc_single_sim, ".", label=f"Kemeny Adjusted", color=colors[method1], alpha=0.5, mew=0)
plt.plot(lambda2s, (k_preds[:,method2]-kc_single_sim)/kc_single_sim, ".", label="Corrected", color=colors[method2], alpha=0.5, mew=0)
plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Percolation Time Prediction Error \n (Single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")
plt.ylim([-1,1])


err_baseline = np.abs((k_preds[:,0]-kc_single_sim)/kc_single_sim)
err_1 = np.abs((k_preds[:,method2]-kc_single_sim)/kc_single_sim)

plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
plt.plot(lambda2s, err_1-err_baseline, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Pairwise \n (single Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")
plt.ylim([-1,1])

plt.figure(dpi=800)
plt.hlines(0, xmin=0, xmax=1,linestyle="--", color="grey")
plt.plot(lambda2s, (k_pred_all[:,0]-kc_all_sim)/kc_all_sim, ".", label=f"Independent Model", color="black", alpha=0.5, mew=0)
plt.plot(lambda2s, (k_pred_all[:,method2]-kc_all_sim)/kc_all_sim, ".", color=colors[method2], label=r"$\lambda_2-$Corrected", alpha=0.5, mew=0)
plt.xlabel(r"$\bf{|\lambda_2|}$", fontsize=fontsize, weight="bold")
plt.ylabel(r"$\bf{(K_{c,pred}-K_c)/K_c}$", fontsize=fontsize, weight="bold")
plt.legend(fontsize=fontsize,prop={'weight':"bold"})
plt.title("Relative Percolation Time Prediction Error \n (All Source)", fontsize=fontsize, weight="bold")
plt.xticks(fontsize=fontsize, weight="bold")
plt.yticks(fontsize=fontsize, weight="bold")

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
