# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 23:36:43 2025

@author: hibado
"""

import numpy as np
# import networkx as nx
# from itertools import combinations
# import matplotlib.pyplot as plt 
# import pickle 
import matplotlib.pyplot as plt
from itertools import combinations
import math
from scipy.stats import dirichlet

num_regions = 100
num_trial = 100000
p_arr = []
num_robot= 3

p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
    
K = 10

#%% 2 robots 
num_robot= 2 
K=10
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

S2 = np.sum(p**2)

kc = np.ones(num_trial)*np.nan

for i in range(num_trial):
    state = np.ones(3, dtype=bool)*False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        if r[0]==r[1]:
            kc[i] = k
            break

p_preds=[]
for T in np.arange(1,K+1,1):
    p_pred= 1-(1-S2)**T
    p_preds.append(p_pred.copy())
k_arr = []
for i in range(num_trial):
    k = np.zeros(K)

    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0),color="red", label="trial")
plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue", label="predicted")
plt.legend()
#%% 0 not meet 1, 1  not meet 2
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

for i in range(num_trial):
    state = True 
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        state *= (not r[0]==r[1]) & (not r[1]==r[2])
    states.append(state)
    
print(sum(states)/num_trial)

print(np.inner(p, (1-p)**2)**K)

#%% 0  met 1, 0  met 2, 1 met 2
K=10
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

S2 = np.sum(p**2)
S3 = np.sum(p**3)
a = 1-2*S2+S3

kc = np.ones(num_trial)*np.nan

for i in range(num_trial):
    state = np.ones(3, dtype=bool)*False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        state[0] += r[0]==r[1]
        state[1] += r[0]==r[2]
        state[2] += r[1]==r[2]
        if state.all():
            kc[i] = k
            break
    states.append(state.all())

p_preds=[]
for T in np.arange(1,K+1,1):
    p_pred= 1- 3*((1-S2)**T) + 3*a**T - (1-3*S2+2*S3)**T
    p_preds.append(p_pred.copy())
k_arr = []
for i in range(num_trial):
    k = np.zeros(K)

    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0),color="red" )
plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue")

#%% nobody meets
K=10
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

S2 = np.sum(p**2)
S3 = np.sum(p**3)

for i in range(num_trial):
    state = True
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        state *= (not r[0]==r[1]) & (not r[1]==r[2]) & (not r[0]==r[2])
    states.append(state)
    
print(sum(states)/num_trial)

print((1-3*S2+2*S3)**K)

#%% 0 meet 1 or 0 meet 2 at least once
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
K=5


for i in range(num_trial):
    rs=[]
    state = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        if r[0]==r[1] or r[0]==r[2]:
            state = True
            break 
    states.append(state)
print(sum(states)/num_trial)

S2 = np.inner(p,p)
S3  = np.sum(p**3)
a = 1-2*S2+S3
print(1-a**K)
 
#%% 0 not meet 1, 1 met 2 at least once

states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
K=3
for i in range(num_trial):
    rs=[]
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs.append(r)
    rs = np.array(rs)
    state = (not( rs[:,0] == rs[:,1]).any()) and ((rs[:,1] == rs[:,2]).any())
    states.append(state)
print(sum(states)/num_trial)

S2 = np.inner(p,p)
S3  = np.sum(p**3)
print((1-np.inner(p,p))**K-(1-2*S2+S3)**K)

#%% 0 met 2 only after 2 met 1
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
ptp = np.inner(p,p)
K=10
S2 = np.inner(p,p)
S3  = np.sum(p**3)
a = 1-2*S2+S3
b = S2 - S3
d = 1-S2
p_pred= 0
for k in np.arange(1,K+1,1):
    p_pred += a**(k-1)*b*(1-d**(K-k))
    
for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r
        if (r[0] == r[2]) and (~(rs[:k,1] == rs[:k,2])).all():
            break
        if (r[0] == r[2]) and (rs[:k,1] == rs[:k,2]).any(): 
            Q = True
            break 
    rs = np.array(rs)
    state = Q
    states.append(state)   
    
print(sum(states)/num_trial)
print(p_pred)

#%% 0 met 2 at least once after 2 met 1
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
ptp = np.inner(p,p)
K=10
S2 = np.inner(p,p)
S3  = np.sum(p**3)
a = 1-2*S2+S3
b = S2 - S3
p_pred= 0
for k in np.arange(1,K+1,1):
    p_pred += ((1-S2)**(k-1))*S2*(1-(1-S2)**(K-k))
    
for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r

        if (r[0] == r[2]) and (rs[:k,1] == rs[:k,2]).any(): 
            Q = True
            break 
    state = Q
    states.append(state)   
    
print(sum(states)/num_trial)
print(p_pred)

#%%  0 met 2 at least once after or during 2 met 1
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
K=10
S2 = np.inner(p,p)
S3  = np.sum(p**3)
a = 1-2*S2+S3
b = S2 - S3
p_pred= 0
for k in np.arange(1,K+1,1):
    p_pred += ((1-S2)**(k-1))*(S2-S3)*(1-(1-S2)**(K-k))+((1-S2)**(k-1)*S3)
    
for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r

        if (r[0] == r[2]) and (rs[:k+1,1] == rs[:k+1,2]).any(): 
            Q = True
            break 
    state = Q
    states.append(state)   
    
print(sum(states)/num_trial)
print(p_pred)


#%% 0 not meet 1, 0 met 2 after 2 met 1
K = 10
states=[]
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
S2 = np.sum(p**2)
S3  = np.sum(p**3)
p_pred= 0
a = 1-2*S2+S3
for k in np.arange(1,K+1,1):
    p_pred += ((a)**(k-1))*(S2-S3)*((1-S2)**(K-k)-a**(K-k))


for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r
        # if r[0]==r[1]:
        #     Q = False
        #     break
        if (r[0] == r[2]) and (rs[:k,1] == rs[:k,2]).any(): 
            Q = True
             
    Q = Q * (not (rs[:,0] == rs[:,1]).any())
    states.append(Q)   
    
print(sum(states)/num_trial)
print(p_pred)

#%% (0  meet 1) or (0 met 2 after 2 met 1)
K = 10
states=[]
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
S2 = np.sum(p**2)
S3  = np.sum(p**3)
p_pred= 0
S2 = np.sum(p**2)
S3  = np.sum(p**3)
p_pred= 0
a = 1-2*S2+S3
for k in np.arange(1,K+1,1):
    p_pred += ((a)**(k-1))*(S2-S3)*((1-S2)**(K-k)-a**(K-k))

p_pred += 1-(1-S2)**K

for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r

        if (r[0] == r[2]) and (rs[:k,1] == rs[:k,2]).any(): 
            Q = True
             
    Q = Q + ((rs[:,0] == rs[:,1]).any())
    states.append(Q)   
    
print(sum(states)/num_trial)
print(p_pred)

#%% 0 meet 1 at least once on or after (1 met 2 and 0 met 2)
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
states = []
K=3
S2 = np.sum(p**2)
S3 = np.sum(p**3)
a = 1-2*S2+S3
b = S2 - S3
p_pred= 0
for k in np.arange(1,K+1,1):
    p_pred += 2*(1-S2)**(k-1)*(S2-S3)*(1-(1-S2)**(K-k)) + 2*(1-S2)**(k-1)*(S3)  
    p_pred -= 2*a**(k-1)*(S2-S3)*(1-(1-S2)**(K-k)) + a**(k-1)*(S3)
    
for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r
        if (r[0] == r[1]) and ((rs[:k+1,1] == rs[:k+1,2]).any() and (rs[:k+1,0] == rs[:k+1,2]).any()): 
            Q = True
            break 
    states.append(Q)   
    
print(sum(states)/num_trial)
print(p_pred)


#%% (0 met 2 after or during 2 met 1) or (0 met 2 after or during 0 met 1) 
K = 10
states=[]
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

S2 = np.sum(p**2)
S3  = np.sum(p**3)
p_pred= 0
a = 1-2*S2+S3
for k in np.arange(1,K+1,1):
    p_pred +=  2*a**(k-1)*(S2-S3)*(1-(1-S2)**(K-k)) + a**(k-1)*(S3)

for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = False
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r
        if ((r[0] == r[2]) and ((rs[:k+1,1] == rs[:k+1,2]).any() or (rs[:k+1,0] == rs[:k+1,1]).any())): 
            Q = True
            break  
    states.append(Q)   
    
print(sum(states)/num_trial)
print(p_pred)

#%% 1 met 2 at least once after 0 meet 2 and 0 met 2 at least once after 1 meet 2
K = 10
states=[]
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

S2 = np.sum(p**2)
S3  = np.sum(p**3)
p_pred= 0
a = 1-2*S2+S3

p_pred= 0
for k in np.arange(1,K+1,1):
    p_pred += 2*(1-S2)**(k-1)*(S2-S3)*(1-(1-S2)**(K-k)) + 2*(1-S2)**(k-1)*(S3)  
    p_pred -= 2*a**(k-1)*(S2-S3)*(1-(1-S2)**(K-k)) + a**(k-1)*(S3)
for i in range(num_trial):
    rs=np.zeros((K, 3))
    Q = np.array([False, False])
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        rs[k,:] = r
        # if(r[0]==r[1] and r[1]==r[2]):
        #     break
        if ((r[1] == r[2]) and (rs[:k+1,0] == rs[:k+1,2]).any()): 
            Q[0] = True
        if ((r[0] == r[2]) and (rs[:k+1,1] == rs[:k+1,2]).any()): 
            Q[1] = True  
        if (Q.all()):
            break
    states.append(Q.all())   
    
print(sum(states)/num_trial)
print(p_pred)
#%% (1 met 2 at least once after 0 meet 2) and (0 met 2 at least once after 1 meet 2) and (0 never meet 1)
K=20
states = []
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

S2 = np.sum(p**2)
S3 = np.sum(p**3)
a = 1-2*S2+S3

k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    rs=np.zeros((K, 3))
    state = np.ones(3, dtype=bool)*False
    state[2] = True
    for t in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        if(r[0]==r[1]):
            state[2] = False
            break
        rs[t,:] = r
        state[0] += (r[1]==r[2]) and (rs[:t,0] == rs[:t,2]).any()
        state[1] += (r[0]==r[2]) and (rs[:t,1] == rs[:t,2]).any()
        k[t] = state.all()
        
    states.append(state.all())
    k_arr.append(k)
p_preds=[]
for T in np.arange(1,K+1,1):
   p_pred= 0
   for t in np.arange(1,T+1,1):
     p_pred+= 2*(a**(t-1)-(1-3*S2+2*S3)**(t-1))*(S2-S3)*((1-S2)**(T-t)-a**(T-t))
   p_preds.append(p_pred.copy())
# for i in range(num_trial):
#     k = np.zeros(K)

#     if not np.isnan(kc[i]):
#         k[int(kc[i]):]=1
#     k_arr.append(k)    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0),color="red" )
plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue")

#%% single source (0) percolation (3-robot) 
K = 20
num_robot= 3

states=[]
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
S2 = np.sum(p**2)
S3  = np.sum(p**3)
a = 1-2*S2+S3

kc = np.ones(num_trial)*np.nan
for trial in range(num_trial):
    rs=np.zeros((K, 3))
    state = [True,0,0]
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        state[1] = state[1] or (r[1] == r[0]) or (r[1] == r[2] and (state[2]))
        state[2] = state[2] or (r[2] == r[0]) or (r[1] == r[2] and (state[1]))
        rs[k,:] = r

        if (np.prod(state)):
            kc[trial]= k
            break
    states.append(np.prod(state))   


p_preds=[]
for T in np.arange(1,K+1,1):
    p_pred= 0
    for t in np.arange(1,T+1,1):
        p_pred += 2*((a)**(t-1))*(S2-S3)*((1-S2)**(T-t)-a**(T-t))

    p_pred += 2*(1-(1-S2)**T) - (1-a**T)
    p_preds.append(p_pred.copy())
k_arr = []
for i in range(num_trial):
    k = np.zeros(K)

    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0),color="red", label="trials" )
plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue", label="Predicted")
plt.legend()

#%% multi-source percolation
from itertools import combinations
num_regions = 20
K = 50
states=[]
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
p = np.ones(num_regions)
p = p/sum(p)
S2 = np.sum(p**2)
S3  = np.sum(p**3)
a = 1-2*S2+S3


# p_pred += 2*(1-(1-S2)**K) - (1-a**K)
kc = np.ones(num_trial)*np.nan
for trial in range(num_trial):
    print(str(trial)+"/"+str(num_trial))
    state = np.eye(3, dtype=bool)
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                state[i,:] = state[i,:] | state[j,:]
                state[j,:] = state[i,:] | state[j,:]
                
        if (np.prod(state)):
            kc[trial]= k
            break
    states.append(np.prod(state[0:2,:]))   
    

k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
p_preds=[]
for T in np.arange(1,K+1,1):
    p_pred= 0
    for t in np.arange(1,T+1,1):
        p_pred += 2*(a**(t-1)-(1-3*S2+2*S3)**(t-1))*(S2-S3)*((1-S2)**(T-t)-a**(T-t))
    p_pred= 3*p_pred
    p_pred+=1- 3*((1-S2)**T) + 3*a**T - (1-3*S2+2*S3)**T


    p_preds.append(p_pred.copy())

plt.figure(dpi=800)
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red", label="trials" )
plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue", label="Predicted" )
plt.plot((1-(1-S2)**(1+np.arange(K))))

plt.xlabel("K (time step)")
plt.ylabel("percolation probability")
plt.title("3 agent percolation")
plt.legend()

#%% single-source percolation N robot 
num_trial = 1000
num_robot = 100
num_regions = 50

states=[]
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
# p = np.ones(num_regions)
# p = p/sum(p)
p = dirichlet.rvs(0.5*np.ones(num_regions))[0]

S2 = np.sum(p**2)

# prediction
M = np.zeros((num_robot,num_robot))
for i in range(num_robot):
    for j in range(num_robot):
        #j: informed 
        I = j+1
        n = num_robot - I # number of uninformed robot
        r = i - j  #increment of informed robot 
        if i>=j:
            M[i,j] = math.comb(n,r)*(1-(1-S2)**I)**r*((1-S2)**I)**(n-r)
Q = M[:num_robot-1, :num_robot-1]
v = np.zeros(num_robot-1)
v[0]=1
k_pred = np.ones(num_robot-1)@np.linalg.inv(np.eye(num_robot-1)-Q)@v

K =int(3*k_pred)

percentage_percolation= np.ones(num_trial)*np.nan
kc = np.ones(num_trial)*np.nan
for trial in range(num_trial):
    print(str(trial)+"/"+str(num_trial))
    state = np.eye(num_robot, dtype=bool)
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                state[i,:] = state[i,:] | state[j,:]
                state[j,:] = state[i,:] | state[j,:]
        if k>= k_pred:
            percentage_percolation[trial] = np.nanmin((percentage_percolation[trial], sum(state[:,0])/num_robot))
        if (np.prod(state[:,0])):
            kc[trial]= k
            if k<k_pred:
                percentage_percolation[trial] = 1
            break
    

#average percolation probability
k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    

percentage_percolation_mean = np.mean(percentage_percolation)
print(f"average percolation percentage = {percentage_percolation_mean}")

plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red" )
# plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue")
plt.xlabel("K (time step)")
plt.ylabel("percolation probability")
# plt.vlines(np.log(num_robot)/((num_robot)*S2),0,1)
plt.vlines(k_pred,0,1, label="Mean Passage Time", color="blue", linestyle="--")
plt.vlines(np.nanmean(kc),0,1, label="mean percolation time", color="red", linestyle="-")
plt.vlines(np.nanmean(kc)+np.nanstd(kc),0,1, label="STD", color="red", linestyle="--")
plt.vlines(np.nanmean(kc)-np.nanstd(kc),0,1, color="red", linestyle="--")

plt.legend()
plt.title(f"single source percolation ({num_robot} agents)")
#%% multi-source percolation N robot 

num_robot = 100

states=[]
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
p = dirichlet.rvs(1*np.ones(num_regions))[0]

S2 = np.sum(p**2)
S3  = np.sum(p**3)
a = 1-2*S2+S3
# prediction
M = np.zeros((num_robot,num_robot))
for i in range(num_robot):
    for j in range(num_robot):
        #j: informed 
        I = j+1
        n = num_robot - I # number of uninformed robot
        r = i - j  #increment of informed robot 
        if i>=j:
            M[i,j] = math.comb(n,r)*(1-(1-S2)**I)**r*((1-S2)**I)**(n-r)
Q = M[:num_robot-1, :num_robot-1]
v = np.zeros(num_robot-1)
v[0]=1
k_pred = np.ones(num_robot-1)@np.linalg.inv(np.eye(num_robot-1)-Q)@v

K =int(2*k_pred)


kc = np.ones(num_trial)*np.nan
for trial in range(num_trial):
    print(str(trial)+"/"+str(num_trial))
    state = np.eye(num_robot, dtype=bool)
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                state[i,:] = state[i,:] | state[j,:]
                state[j,:] = state[i,:] | state[j,:]
                
        if (np.prod(state)):
            kc[trial]= k
            break
    states.append(np.prod(state))   
    

k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    


    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red" )
# plt.plot(np.arange(1,K+1,1),p_preds,"--",color="blue")
plt.xlabel("K (time step)")
plt.ylabel("percolation probability")
plt.vlines(k_pred,0,1, label="Mean Passage Time")
plt.legend()
plt.title(str(num_robot)+" agent percolation")

#%% only one pair in N robot meet
num_robot=10
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
states=[]
for i in range(num_trial):
    r = np.random.choice(range(num_regions), num_robot, p=p)
    states.append(sum(r==r[0])==2)
print(sum(states)/num_trial)
print((num_robot-1)*np.sum([ (pi**2)*(1-pi)**(num_robot-2) for pi in p]))


#%% n robot meet 0 at k
num_robot=9
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
states=[]
n = 2
for i in range(num_trial):
    r = np.random.choice(range(num_regions), num_robot, p=p)
    states.append(sum(r==r[0])==(n+1))
print(sum(states)/num_trial)
print((math.comb(num_robot-1, n))*np.sum([(pi)**(n+1)*(1-pi)**(num_robot-1-n) for pi in p]))

#%% n robot meet 0 within K
num_robot=9
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
states=[]
n = num_robot
K=20                                                                                                                                                                                                        
for i in range(num_trial):
    state = False
    for k in range(20):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        state+= 0
    states.append(sum(r==r[0])==(n+1))
print(sum(states)/num_trial)
print((math.comb(num_robot-1, n))*np.sum([(pi)**(n+1)*(1-pi)**(num_robot-1-n) for pi in p]))
#%% n robot meet 0 or 1
num_robot=9
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
S2 = np.sum(p**2)

states=[]
n = 3
k = 2
for i in range(num_trial):
    r = np.random.choice(range(num_regions), num_robot, p=p)
    if r[0]==r[1]:
        states.append((sum((r==r[0])[k:])==n))
    else:
        states.append(((sum((r==r[0])[k:])+(sum((r==r[1])[k:])))==n))
print(sum(states)/len(states))
pred = 0
for m in np.arange(1,k+1):
    for idx in combinations(range(num_regions), m): 
        p_sum=np.sum(p[np.array(idx)])
        pred+= (p_sum**k-1)*(p_sum)**(n)*(1-p_sum)**(num_robot-k-n)
print((math.comb(num_robot-k, n))*pred)
#print(pred)
