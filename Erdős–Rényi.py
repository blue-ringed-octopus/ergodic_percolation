# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 18:40:09 2025

@author: hibado
"""

import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt 
n = 50

critical=[]

connected = [[] for _ in range(30)]
for p in np.linspace(0,0.5,100):
    for k in range(30):
        G=nx.Graph()
        for i in range(n):
            G.add_node(str(i))
            
        for i,j in combinations(range(n), 2): 
            if np.random.rand()<p:
                G.add_edge(str(i), str(j))
        connected[k].append(nx.is_connected(G))
#%%
connected =np.array(connected)
connected_mean = np.mean(connected, axis=0)
P_c = np.log(n)/n

plt.plot(np.linspace(0,0.5,100), connected_mean,'.')
plt.vlines(P_c, 0, 1, linestyle="--", color="r")
