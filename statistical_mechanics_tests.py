# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:58:58 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

from tqdm import tqdm

num_regions = 10
num_robot = 5
num_trial = 90000

rho0 = dirichlet.rvs(np.ones(num_regions))[0]
P = dirichlet.rvs(np.ones(num_regions), num_regions).T


r0 = np.random.choice(range(num_regions), size=num_robot, p= rho0) 
r0 = r0.astype(np.int32)
n0 = np.zeros(num_regions)
for r in r0:
    n0[r] += 1

K = np.zeros((num_trial,num_regions,num_regions), dtype=np.int32)
for trial in tqdm(range(num_trial)):
    r1 = np.zeros(num_robot, dtype=np.int32)   
    for robot in range(num_robot):
        r1[robot] = np.random.choice(range(num_regions), p=P[:,r0[robot]])   
        K[trial, r1[robot],  r0[robot] ]+=1


n1 = np.zeros((num_trial, num_regions))
for trial in (range(num_trial)):
    n1[trial] = K[trial]@np.ones(num_regions)

K_mean = np.mean(K, axis=0)
K_pred = P@np.diag(n0)

rho1 =  np.mean(n1, axis=0)
rho1_pred = P@n0
#%% no robot in regoin j
p_no_rob_pred = np.zeros(num_regions)
for j in range(num_regions):
    p_no_rob_pred[j] = np.prod([(1-P[j,i])**n0[i] for i in range(num_regions)])

p_no_rob = np.sum(n1==0, axis=0)/num_trial

#%% not meet at 1 given r0_uninformed
r1_uninformed =  np.zeros(num_trial, dtype=np.int32)
r0_uninformed = np.random.choice(range(num_regions), p= rho0)
for trial in tqdm(range(num_trial)):
    r1_uninformed[trial] = np.random.choice(range(num_regions), p=P[:,r0_uninformed])

P_not_meet_at_j = [P[j, r0_uninformed]*p_no_rob[j] for j in range(num_regions)]
P_not_meet_pred = sum(P_not_meet_at_j)    
P_not_meet = sum([n1[trial, r1_uninformed[trial]] == 0 for trial in range(num_trial)])/num_trial
#%% not meet at 1 random, any r0_uninformed
r0_uninformed =  np.zeros(num_trial, dtype=np.int32)
r1_uninformed =  np.zeros(num_trial, dtype=np.int32)

for trial in tqdm(range(num_trial)):
    r0_uninformed[trial] = np.random.choice(range(num_regions), p= rho0)
    r1_uninformed[trial] = np.random.choice(range(num_regions), p=P[:,r0_uninformed[trial]])


P_not_meet_marginal_pred = sum([rho0[i]*sum([ P[j, i]*p_no_rob[j] for j in range(num_regions)]) for i in range(num_regions) ])
P_not_meet_marginal  = sum([n1[trial, r1_uninformed[trial]] == 0 for trial in range(num_trial)])/num_trial

print(f"err (not meet marginal): {P_not_meet_marginal_pred-P_not_meet_marginal}")
#%% not meet at 1 given not meet at 0, given n0
r0_uninformed =  np.zeros(num_trial, dtype=np.int32)
r1_uninformed =  np.zeros(num_trial, dtype=np.int32)
rho0_conditional = rho0*(n0 == 0)
rho0_conditional /= sum(rho0_conditional)
r0 = np.random.choice(range(num_regions), size=num_robot, p= rho0) 
r0 = r0.astype(np.int32)
n0 = np.zeros(num_regions)
for r in r0:
    n0[r] += 1
    
for trial in tqdm(range(num_trial)): 
    r0_uninformed[trial] = np.random.choice(range(num_regions), p= rho0_conditional)
    r1_uninformed[trial] = np.random.choice(range(num_regions), p=P[:,r0_uninformed[trial]])
    
P_not_meet_conditional = sum([n1[trial, r1_uninformed[trial]] == 0 for trial in range(num_trial)])/num_trial
P_not_meet_conditional_pred = sum([rho0_conditional[i]*sum([ P[j, i]*p_no_rob[j] for j in range(num_regions)]) for i in range(num_regions) ])

#%% not meet at 1 given not meet at 0
r0_uninformed =  np.zeros(num_trial, dtype=np.int32)
r1_uninformed =  np.zeros(num_trial, dtype=np.int32)



n1 = np.zeros((num_trial, num_regions))
   
K = np.zeros((num_trial,num_regions,num_regions), dtype=np.int32)
for trial in tqdm(range(num_trial)):
    n0 = np.ones(num_regions)
    while sum(n0==0) == 0:
        n0 = np.zeros(num_regions)
        r0 = np.random.choice(range(num_regions), size=num_robot, p= rho0) 
        r0 = r0.astype(np.int32)
        for r in r0:
            n0[r] += 1
    rho0_conditional = rho0*(n0 == 0)
    rho0_conditional /= sum(rho0_conditional)        
    r1 = np.zeros(num_robot, dtype=np.int32)   
    for robot in range(num_robot):
        r1[robot] = np.random.choice(range(num_regions), p=P[:,r0[robot]])   
        K[trial, r1[robot],  r0[robot] ]+=1
        
    n1[trial] = K[trial]@np.ones(num_regions)
    r0_uninformed[trial] = np.random.choice(range(num_regions), p= rho0_conditional)
    r1_uninformed[trial] = np.random.choice(range(num_regions), p=P[:,r0_uninformed[trial]])

P_not_meet_conditional2 = sum([n1[trial, r1_uninformed[trial]] == 0 for trial in range(num_trial)])/num_trial
P_not_meet_at_0 = sum([rho*(1-rho) for rho in rho0])
P_not_meet_conditional2_pred = sum([rho0[k]/(1-rho0[k])*sum([rho0[i]*sum([P[j,i]*(1-P[j,k])for j in range(num_regions)]) for i in range(num_regions) if not i==k]) for k in range(num_regions)])

# P_not_meet_conditional2_pred = (1/P_not_meet_at_0)*sum([rho0[k]*sum([rho0[i]*sum([P[j,i]*(1-P[j,k])for j in range(num_regions)]) for i in range(num_regions) if not i==k]) for k in range(num_regions)])
P_not_meet_conditional2_pred = P_not_meet_conditional2_pred**num_robot
# P_not_meet_conditional2_pred = 1- P_not_meet_conditional2_pred