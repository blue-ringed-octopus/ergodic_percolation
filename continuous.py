# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 00:40:16 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from scipy.stats import multivariate_normal as norm
from scipy.interpolate import LinearNDInterpolator

sigma = 0.2
num_robots = 50
class Agent:
    def __init__(self, id_):
        self.id= id_
        self.reset()
    
    def reset(self):
        self.x = np.random.rand(2)
        self.info = np.zeros(num_robots, dtype=bool)
        self.info[self.id] = 1
        
    def step(self, target):
        dx = norm.rvs(self.x, sigma)
        x_prime = self.x + dx
        if (x_prime[0]>=w) or (x_prime[0]<0) or (x_prime[1]>=h) or (x_prime[1]<0):
            A = 0
        else:
            A = 1
        if np.random.rand()<=A:
            self.x = x_prime

        
        
w, h = 1,1
radius = 0.05
agents=[]
for i in range(num_robots):
    agents.append(Agent(i))
    
T = 100
num_trial = 1000
informed = np.ones((num_trial, T))
for trial in tqdm(range(num_trial)):
    for agent in agents:
        agent.reset()
    for t in range(T):
        for i,j in combinations(range(num_robots), 2): 
            if np.linalg.norm(agents[i].x-agents[j].x)<radius:
                info = agents[i].info.copy() |  agents[j].info.copy()
                agents[i].info = info
                agents[j].info = info
        informed[trial,t] = np.sum([agents[i].info[0] for i in range(num_robots)])/num_robots
        if informed[trial,t] == 1:
            break
        for agent in agents:
            agent.step(None)
        
plt.figure()
plt.plot(np.mean(informed[:, :50], axis=0))

p_informed = informed.copy()
p_informed[p_informed<1] = 0 
plt.figure()
plt.plot(np.mean(p_informed[:, :50], axis=0))
