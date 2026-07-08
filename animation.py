# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 14:05:58 2025

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt 
from graph_planner import REMC
from A_star import A_star
from map_manager import Map_Manager
from itertools import combinations
import networkx as nx 
import math 
num_agents = 30


manager = Map_Manager()
graph=manager.hierarchical_graph.levels[1]
#%%
p = np.ones(graph.n)
p = p/sum(p)

_, edges = manager.get_graph(1)
P = REMC(p, edges)

S2 = np.sum(p**2)

K = 10
M = np.zeros((num_agents,num_agents))
for i in range(num_agents):
    for j in range(num_agents):
        #j: informed 
        I = j+1
        n = num_agents - I # number of uninformed robot
        r = i - j  #increment of informed robot 
        if i>=j:
            M[i,j] = math.comb(n,r)*(1-(1-S2)**I)**r*((1-S2)**I)**(n-r)
Q = M[:num_agents-1, :num_agents-1]
v = np.zeros(num_agents-1)
v[0]=1
k_pred = np.ones(num_agents-1)@np.linalg.inv(np.eye(num_agents-1)-Q)@v


    
path_planner = A_star(manager.masked_region_map.mask.T)
k_arr = []
targets=[[] for _ in range(num_agents)]
r = np.random.choice(range(graph.n), num_agents, p=p)
rs = [r]
x = [[] for _ in range(num_agents)]
for j in range(num_agents):
    id_ = np.random.choice(list(graph.nodes[r[j]].children.keys()))
    x[j].append(np.array(graph.nodes[r[j]].children[id_].coord))
state = np.eye(num_agents, dtype=bool)
informed_percentage = np.ones(K)
# informed = [state[0,:].copy()]
informed=[]
connectivity = np.zeros((K,num_agents,num_agents))
for k in range(K):
    for i,j in combinations(range(num_agents), 2): 
        if r[i]==r[j]:
            info = state[i,:].copy() |  state[j,:].copy() 
            state[i,:] = info 
            state[j,:] = info 
            connectivity[k,i,j] = True
            connectivity[k,j,i] = True

    informed.append(state[:,0].copy())
    informed_percentage[k] = np.sum(state[:,0].copy())/num_agents
    for robot in range(num_agents):
        r[robot] = np.random.choice(range(graph.n), 1, p=P[:,r[robot]])
        
    for j in range(num_agents):
        path = []
        while len(path)==0:
            id_ = np.random.choice(list(graph.nodes[r[j]].children.keys()))
            target = graph.nodes[r[j]].children[id_].coord
         #   path, _ = path_planner.plan([x[j][-1][1], x[j][-1][0]], [target[1], target[0]])
            path, _ = path_planner.plan(x[j][-1], target)
            
        targets[j].append(target)    
        x[j] += path
    max_len = max([len(path) for path in x]) 
    for j in range(num_agents):
        for _ in range(max_len-len(x[j])):
            x[j].append(x[j][-1])
    k_arr += [k for _ in range(max_len - len(k_arr))]
    
    
    rs.append(r.copy())  

states = np.arange(1,num_agents+1)/num_agents
#%%
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

step = 2
x = np.array(x)
targets = np.array(targets)

fig, ax = plt.subplots(1, 3,layout="constrained", dpi = 800)

ax[0].set_aspect('equal', adjustable='box')
# ax[1].set_aspect('equal', adjustable='box')

ax[0].tick_params(axis='x', length=0)  # Hide x-axis ticks
ax[0].xaxis.set_ticklabels([])
ax[0].tick_params(axis='y', length=0)  # Hide y-axis ticks
ax[0].yaxis.set_ticklabels([])

def get_agent_graph(k):
    G=nx.Graph()
    G.add_nodes_from(range(num_agents))
    for i,j in combinations(range(num_agents), 2): 
        if connectivity[k,i,j]:
            G.add_edge(i,j)
    return G

def animate(i):
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    t = i*step
    k = k_arr[t]

    im = ax[0].imshow(manager.masked_region_map, cmap="Greys_r")
    node_color = []
    for robot in range(num_agents):  
        if informed[k][robot]:
            node_color.append("red")
            ax[0].plot(x[robot,t,1],x[robot,t,0], ".", color = "red")
        else:
            ax[0].plot(x[robot,t,1],x[robot,t,0], ".", color = "blue")
            node_color.append("blue")
  

    ax[0].tick_params(axis='x', length=0)  # Hide x-axis ticks
    ax[0].xaxis.set_ticklabels([])
    ax[0].tick_params(axis='y', length=0)  # Hide y-axis ticks
    ax[0].yaxis.set_ticklabels([])
   
    ax[0].set_aspect('equal', adjustable='box')
    # ax[1].set_aspect('equal', adjustable='box')
    ax[0].set_title("Region Graph")
 #   ax[1].set_title("True Confidence")
    ax[1].set_title("Comm. Network")
    ax[1].set_aspect('equal', adjustable='box')
    plt.subplot(1,3,2)
    G = get_agent_graph(k)
    # nx.draw(G, pos=nx.circular_layout(G), node_color=node_color,node_size=10)
    pos = nx.circular_layout(G)
    nodes = np.array([pos[v] for v in G])
    edges = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    edge_color = ["red" if(informed[max(k-1,0)][u] ^ informed[max(k-1,0)][v]) else "gray"  for u, v in G.edges()]
    for i, vizedge in enumerate(edges):
            ax[1].plot(*vizedge.T, color=edge_color[i], linewidth="1")
    ax[1].scatter(*nodes.T, alpha=1, s=50, color=node_color, zorder=3)
    ax[1].set_axis_off()

    ax[2].set_title("Informed Percentage")
    ax[2].plot(informed_percentage[0:k+1])
    ax[2].vlines(np.ceil(k_pred),0,1, label="Mean Passage Time", color="gray", linestyle="--")

    ax[2].set_ylim([0,1])
    ax[2].set_xlim([0,K])
    ax[2].set_xlabel("k" )
    # plt.show()
    return [im]

animate(100)

#%%
step = 2
ani = FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=len(x[0])//step)    
ani.save("test.mp4", dpi=500,  writer=FFMpegWriter(fps=25, bitrate=9999))     

# ani.save(planner+"_"+method+"_"+str(num_agents)+".mp4", dpi=500,  writer=FFMpegWriter(fps=25, bitrate=1000))     
