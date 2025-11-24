# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:30:15 2025

@author: hibado
"""

import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt 
n = 30
K = 50
critical=np.zeros([20,30])*np.nan
p_c = np.log(n)/n
p_arr = np.linspace(0.001,0.2, 20)
for m in range(len(p_arr)):
    p = p_arr[m]
    for trial in range(30):
        G=nx.DiGraph()
        for k in range(K):
            reachable = 1
            for i in range(n):
                G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k+1)+"]")
                G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
    
            for i,j in combinations(range(n), 2): 
                if np.random.rand()<p:
                    G.add_edge(str(i)+"["+str(k)+"]", str(j)+"["+str(k)+"]")
                    G.add_edge(str(j)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
            for i in range(n):
                for j in range(n):
                    if not j==i:
                        reachable = reachable*nx.has_path(G,str(i)+"["+str(0)+"]", str(j)+"["+str(k)+"]")
            if reachable:
                critical[m,trial]=k
                break
#%%
stat= np.nanmean(critical, axis=1)
plt.plot(p_arr, critical, '.', color="r", alpha=0.05)
plt.plot(p_arr, stat, '--')
# plt.plot(p_arr , np.log(1-p_c)/np.log(1-p_arr), '--',color="r")

plt.vlines(p_c,0,np.nanmax(critical) ,linestyle = "--", label="bernoulli percolation")
plt.xlabel("p")
plt.ylabel("k")
plt.yticks(np.arange(0,int(np.nanmax(critical)),3))
plt.legend()