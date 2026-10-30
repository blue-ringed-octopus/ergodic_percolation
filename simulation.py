# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:16:37 2026

@author: hibado
"""
import numpy as np
import networkx as nx

class Agent:
    def __init__(self, id_):
        self.id = id_
        self.reset()
        
    def reset(self):
        self.information = {self.id}
        
        
    # def set_r(self,r):
    #     self.r =r 
        
    def move(self, P):
        self.r = np.random.choice(range(len(P)), p=P[:,self.r])
        return self.r.copy()
    
    
class World:
    def __init__(self,graph_type, agents):
        self.graph_type = graph_type
        self.agents = agents 
        self.P = None
        
    def set_transition(self, P):
        self.P = P
        
    def reset(self, p0, P):
        self.P = P
        for id_, agent in self.agents.items():
            agent.reset()
            agent.r = np.random.choice(range(len(P)), p=p0)
        
        self.information_state = np.eye(len(self.agents), dtype=bool)
        self.occupancy = {}

    def comunnicate_info(self):
        # for i,j in combinations(self.agents.keys(), 2): 
        #     if r[i]==r[j]:
        #         info = state[i,:].copy() |  state[j,:].copy() 
        #         state[i,:] = info
        #         state[j,:] = info 
        for r, ids in self.occupancy.items():
            info = set()
            if len(ids)>1:
                for id_ in ids:
                    info = info | self.agents[id_].information
                information_state = np.zeros(len(self.information_state), dtype = bool)
                information_state[list(info)] = True
                self.information_state[ids,:] = information_state
                for id_ in ids:
                    self.agents[id_].information = info
                
    def transition_agents(self):
        occupancy = {}
        for id_, agent in self.agents.items():
            r = agent.move(self.P)
            if r not in occupancy.keys():
                occupancy[r]=[]
            occupancy[r].append(id_)
        self.occupancy = occupancy
        
    def step(self):
        if self.P is None:
            raise Exception("transition law not set")
            
        self.transition_agents()
        self.comunnicate_info()
        
        return self.information_state.copy()
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from tqdm import tqdm
    
    num_trial = 100
    K = 1000
    kc = np.ones(num_trial)*np.nan

    h = 30
    w = 30
    num_regions = int(h*w)
    
    diameter = 20
    
    # density = 0.5
    # num_agents = int(num_regions*density)
    num_agents = 10
    p0 = np.ones(num_regions)/num_regions
    g = nx.grid_graph((h,w), periodic = True)
    degree = int(h*w/diameter)
    
    A = nx.adjacency_matrix(g).toarray().T+np.eye(len(g))
    A = np.sum([np.linalg.matrix_power(A,k) for k in np.arange(1,degree)], axis=0).astype(bool)
    P = np.zeros(A.shape)
    for i in range(num_regions):
        idx =  np.where(A[:,i])[0]
        P[idx, i] = np.ones(len(idx))/len(idx)

    agents = {}
    for i in range(num_agents):
        agents[i] = Agent(i)
    world = World("grid",agents)
    for trial in tqdm(range(num_trial)):
        world.reset(p0=p0, P=P)
        for k in range(K):
            information_state = world.step()
            if (np.prod(information_state)):
                kc[trial] = np.nanmin([k,kc[trial]])
                break
    k_arr = []
    for i in range(num_trial):
        k = np.zeros(K)
        if not np.isnan(kc[i]):
            k[int(kc[i]):]=1
        k_arr.append(k)  
    plt.plot(np.mean(k_arr, axis=0)[:int(np.mean(kc)*2)])
    plt.vlines(np.mean(kc),0,1, linestyle= "--")
    plt.title(f"num agents: {num_agents}, std: {np.std(kc)}")