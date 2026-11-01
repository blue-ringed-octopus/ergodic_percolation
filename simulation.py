# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:16:37 2026

@author: hibado
"""
import numpy as np
from tqdm import tqdm

def run_trial(world, P,p0,num_trial, K_max):
    kc_single = np.ones(num_trial)*np.nan
    kc = np.ones(num_trial)*np.nan
    informed_percentage = np.zeros((num_trial,K_max))

    for trial in tqdm(range(num_trial)):
        world.reset(p0=p0, P=P)
        for k in range(K_max):
            informed_percentage[trial, k] = world.get_informed_percentage(0)
            information_state = world.step()
            if (np.prod(information_state[:,0])):
                kc_single[trial]= np.nanmin([k,kc_single[trial]])
                
            if (np.prod(information_state)):
                kc[trial] = np.nanmin([k,kc[trial]])
                informed_percentage[trial, k+1:] = world.get_informed_percentage(0)
                break
    return kc_single, kc, informed_percentage

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
        self.get_occupancy()
        
    
    def get_occupancy(self):
        occupancy = {}
        for id_, agent in self.agents.items():
            r = agent.r
            if r not in occupancy.keys():
                occupancy[r]=[]
            occupancy[r].append(id_)
            
        self.occupancy = occupancy
        
    def comunnicate_info(self):
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
        for id_, agent in self.agents.items():
           agent.move(self.P)
        self.get_occupancy()
        
    def get_informed_percentage(self, info_id):
        I = self.information_state[:,info_id].copy()
        return np.mean(I)
        
        
    def step(self):
        if self.P is None:
            raise Exception("transition law not set")
            
        self.comunnicate_info()    
        self.transition_agents()
        
        return self.information_state.copy()
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from tqdm import tqdm

    from itertools import combinations
    import pickle 

    num_trial = 9999

    output_name = "resources/percolation_trials_pred_complete.pkl"
    with open("resources/percolation_trials_complete.pkl", "rb") as file:
        graph_data = pickle.load(file)

    dat = graph_data[1]
    P = dat["P"]
    num_regions = len(P)
    p0 = dat["p0"]
    num_robot = dat["num_robots"]
    K = 500

    kc_single = np.ones(num_trial)*np.nan
    kc1 = np.ones(num_trial)*np.nan
    rho = np.zeros((num_trial, K, num_regions))
    I_sim = np.zeros((num_trial,K+1, num_regions))
    for trial in tqdm(range(num_trial)):
        state = np.eye(num_robot, dtype=bool)
        r = np.random.choice(range(num_regions), num_robot, p=p0)
        I_sim[trial,0,r[0]] += 1
        for k in range(K): 
            for i,j in combinations(range(num_robot), 2): 
                if r[i]==r[j]:
                    info = state[i,:].copy() |  state[j,:].copy() 
                    state[i,:] = info
                    state[j,:] = info
            # for region in range(num_regions):
            #     if region in r:
            #         robots = np.where(r==region)[0]
            #         state[robots,:] = np.sum([state[rob] for rob in robots], axis=0).astype(bool)
            for robot in range(num_robot):
                I_sim[trial,k+1,r[robot]]+=state[robot,0]
                
            if (np.prod(state[:,0])):
                kc_single[trial]= np.nanmin([k,kc_single[trial]])
                
            if (np.prod(state)):
                I_sim[trial,k+1:,:]= I_sim[trial,k+1,:]
                kc1[trial]= np.nanmin([k,kc1[trial]])
                break
            
            for robot in range(num_robot):
                r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]    
                # r[robot] = np.random.choice(range(num_regions), 1, p=pk[k+1])[0]

                rho[trial,k,r[robot]]+=1/num_robot
            # r[0] = np.random.choice(range(num_regions), 1, p=np.linalg.matrix_power(P, k+1)@I[0])

    rho = np.mean(rho,axis=0)
    k_arr1 = []
    for i in range(num_trial):
        k = np.zeros(K)
        if not np.isnan(kc1[i]):
            k[int(kc1[i]):]=1
        k_arr1.append(k)  

    kc2 = np.ones(num_trial)*np.nan
    agents = {}
    for i in range(num_robot):
        agents[i] = Agent(i)
    world = World(dat["type"],agents)
        
    for trial in tqdm(range(num_trial)):
        world.reset(p0=p0, P=P)
        for k in range(K):
            information_state = world.step()
            if (np.prod(information_state)):
                kc2[trial] = np.nanmin([k,kc2[trial]])
                break
    k_arr2 = []
    for i in range(num_trial):
        k = np.zeros(K)
        if not np.isnan(kc2[i]):
            k[int(kc2[i]):]=1
        k_arr2.append(k)  

    kc_mean = int(np.mean(kc1))
    plt.plot(np.mean(k_arr2, axis=0)[:2*kc_mean])
    plt.plot(np.mean(k_arr1, axis=0)[:2*kc_mean])
