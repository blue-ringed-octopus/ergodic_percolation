# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 23:06:20 2026

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import math
from scipy.stats import dirichlet
from graph_planner import REMC, FMMC
import networkx as nx
from tqdm import tqdm

num_regions = 3
num_trial = 100000
p_arr = []
num_robot= 2
directed = True

# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
p=dirichlet.rvs(1*np.ones(num_regions))[0]

g = nx.erdos_renyi_graph(num_regions, 2*np.log(num_regions)/num_regions, directed = directed)
    
A = nx.adjacency_matrix(g).toarray().T+np.eye(num_regions)

P = FMMC(p, A)
#%% not meet at k=1 given not meet at k=0
p0=dirichlet.rvs(1*np.ones(num_regions))[0]
states=[]
for trial in range(num_trial):
    print(f"{trial}/{num_trial}")
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    while r[0]==r[1]:
        r = np.random.choice(range(num_regions), num_robot, p=p0)

    rs =  [r]
    for robot in range(num_robot):
        r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])
    state=(r[0]==r[1])
    states.append(state)

p_sim=1-sum(states)/len(states)    
p_pred = 1
for i in  range(num_regions):
    for j in range(num_regions):
        p_pred -= P[i,j]*p0[j]/(1-p0[j])*sum([P[i,k]*p0[k] for k in range(num_regions) if not k==j])
print(f"{p_sim}/{p_pred}")
#%% meet at least once in 2 step
p0=dirichlet.rvs(1*np.ones(num_regions))[0]
states=[]
for trial in range(num_trial):
    print(f"{trial}/{num_trial}")
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    rs =  [r]
    state=(r[0]==r[1])
    
    for robot in range(num_robot):
        r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])
    state+=(r[0]==r[1])
    states.append(state)

p_sim=sum(states)/len(states)    
p_pred = 1
for i in  range(num_regions):
    for j in range(num_regions):
        p_pred -= P[i,j]*p0[j]/(1-p0[j])*sum([P[i,k]*p0[k] for k in range(num_regions) if not k==j])

p_pred *= 1-np.inner(p0, p0)
p_pred= 1-p_pred

print(f"{p_sim}/{p_pred}")

#%% meet at least once in K step
def p_conditional(P, p0):
    p_pred = 1
    for i in  range(num_regions):
        for j in range(num_regions):
            p_pred -= P[i,j]*p0[j]/(1-p0[j])*sum([P[i,k]*p0[k] for k in range(num_regions) if not k==j])
    return p_pred


num_regions = 10

p=dirichlet.rvs(1*np.ones(num_regions))[0]

g = nx.erdos_renyi_graph(num_regions, 2*np.log(num_regions)/num_regions, directed = directed)

for i in range(num_regions):
    g.add_edge(i,i)
    
edges = list((g.edges))

P = FMMC(p, edges, directed=directed)
p0=dirichlet.rvs(1*np.ones(num_regions))[0]
states=[]
K=20
kc=np.ones(num_trial)*np.nan
for trial in range(num_trial):
    print(f"{trial}/{num_trial}")
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    rs =  [r]
    state=(r[0]==r[1])
    for k in np.arange(1,K):
        if state:
            kc[trial]=k-1
            break
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])
        state+=(r[0]==r[1])
    states.append(state)

p_sim=sum(states)/len(states)    

p_preds=[]
for T in np.arange(1,K+1,1):
    p_pred = 1-np.inner(p0, p0)
    for k in np.arange(1,T):
        p_pred *= p_conditional(np.linalg.matrix_power(P,k),p0)

    p_pred = 1-p_pred


    p_preds.append(p_pred.copy())


p_preds_independent = [1-(1-np.inner(np.linalg.matrix_power(P,k-1)@p0, np.linalg.matrix_power(P,k-1)@p0))**k for k in np.arange(1,K+1)]


k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red", label="Simulation" )
plt.plot(np.arange(1,K+1,1), p_preds, "--",color = "blue", label="first order correction" )
plt.plot(np.arange(1,K+1,1), p_preds_independent, "--",color = "green", label="Independent" )
plt.legend()
plt.ylim([0,1.1])
plt.title("Meeting Chance in K steps")

#%% probability of being in state 0 at least once (first passage time)
num_regions = 10

p=dirichlet.rvs(1*np.ones(num_regions))[0]

g = nx.erdos_renyi_graph(num_regions, 2*np.log(num_regions)/num_regions, directed = directed)

for i in range(num_regions):
    g.add_edge(i,i)
    
edges = list((g.edges))

P = FMMC(p, edges, directed=directed)
p0=dirichlet.rvs(1*np.ones(num_regions))[0]
states=[]
K=20
kc=np.ones(num_trial)*np.nan
for trial in range(num_trial):
    print(f"{trial}/{num_trial}")
    r = np.random.choice(range(num_regions), 1, p=p0)
    rs =  [r]
    state=(r[0]==0)
    for k in np.arange(1,K):
        if state:
            kc[trial]=k-1
            break
        
        r[0] = np.random.choice(range(num_regions), 1, p=P[:,r[0]])
        state+=(r[0]==0)
    states.append(state)

p_sim=sum(states)/len(states)    

p_preds=[]
Q = P[1:,1:]
q0 = p0[1:]

for T in np.arange(0,K,1):
    p = sum(np.linalg.matrix_power(Q, T)@q0)

    p_preds.append(1-p)



k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red", label="Simulation" )
plt.plot(np.arange(1,K+1,1), p_preds, "--",color = "blue", label="first order correction" )
plt.legend()
plt.ylim([0,1.1])
plt.title("Hitting Chance in K steps")
#%% Collision probability 
num_regions = 10
num_trial = 10000
directed= True

p_hat=dirichlet.rvs(1*np.ones(num_regions))[0]

g = nx.erdos_renyi_graph(num_regions, 2*np.log(num_regions)/num_regions, directed = directed)
if directed:
    g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
    while not nx.is_strongly_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
else:
    g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)
A = nx.adjacency_matrix(g).toarray().T+np.eye(num_regions)

P = FMMC(p_hat, A)
lambda_, _ = np.linalg.eig(P)
lambda_ = np.sort(np.abs(lambda_))
lambda2 = lambda_[-2]

p0=dirichlet.rvs(1*np.ones(num_regions))[0]
# p0 = p_hat

pk = [p0]

 
states=[]
K=20
for k in range(K):
    pk.append(P@pk[-1].copy())
kc=np.ones(num_trial)*np.nan
for trial in tqdm(range(num_trial)):
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    rs =  [r]
    state=(r[0]==r[1])
    for k in np.arange(1,K):
        if state:
            kc[trial]=np.nanmin([k-1,  kc[trial]])
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])[0]
        state+=(r[0]==r[1])
    states.append(state)

p_sim=sum(states)/len(states)    

state_comb =  list(product(range(num_regions), repeat=2))
m = len(state_comb)
P2 = np.zeros((m,m))
p02 =np.zeros(m)
for I in  range(m):
    i1,i2 = state_comb[I]
    p02[I] = p0[i1]*p0[i2]
    for J in range(m):
        j1,j2 = state_comb[J]
        P2[I,J] = P[i1,j1]*P[i2,j2]
        
Q = P2.copy()
q0 = p02.copy()

q1 = np.zeros(len(q0))
collision_idx = []
B = []
for I in  range(m):
    i,j = state_comb[I]
    if i==j:
        collision_idx.append(I)
        Q[:,I] = 0#np.nan
        Q[I,:] = 0#np.nan
        q0[I] = 0#np.nan
        q1[I] = 1
        
B = P2[collision_idx,:].copy()
B = np.delete(B, collision_idx,axis=1)
lambdaQ, V = np.linalg.eig(Q)
idx = np.argmax(np.abs(lambdaQ))
v1 = V[:, idx]

# lambdaQ = np.sort(np.abs(lambdaQ))
lambdaQ_max = np.abs(lambdaQ[idx])
Q_test = Q/np.linalg.norm(Q,1)
np.linalg.matrix_power(Q_test, 99999)
p_preds=[]

for T in np.arange(0,K,1):
    p = sum(np.linalg.matrix_power(Q, T)@q0)


    p_preds.append(1-p.copy())


p_preds_independent = np.array([1-(1-np.inner(pk[k], pk[k]))**k for k in np.arange(1,K+1)])


k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
plt.figure()    
plt.plot(np.arange(1,K+1,1), np.mean(k_arr, axis=0), color = "red", label="Simulation" )
plt.plot(np.arange(1,K+1,1), p_preds, "--",color = "blue", label="First Hitting Model" )
plt.plot(np.arange(1,K+1,1), p_preds_independent, "--",color = "green", label="Independent" )

plt.legend()
plt.ylim([0,1.1])
plt.title("Meeting Chance On Markov Chain in K steps")
plt.xlabel("K (time step)")
plt.ylabel("Pr(meet at least once within K)")
p_preds2=[np.inner(p0,p0)]
for T in np.arange(1,K,1):
    p = q1.T@P2@np.linalg.matrix_power(Q, T-1)@q0
    p_preds2.append(p.copy())
    
plt.figure()
plt.title("First Hitting Chance at K")
plt.plot(np.arange(0,K,1), np.concatenate(([sum(kc==0)/len(kc)],np.diff(np.mean(k_arr,0)))), "blue", label="Simulation")
plt.plot(np.arange(0,K,1), p_preds2,  "red", linestyle ='--', label="First Hitting Model" )
plt.legend()
plt.xlabel("K (time step)")
plt.ylabel("Pr(First Meet Exactly at K)")

p_conditional_pred=[]
for T in range(len(p_preds2)):
    # p_conditional_pred.append(p_preds2[T]/(1-sum(p_preds2[:T])))
    if T == 0:
        p_conditional_pred.append(p_preds2[T])
    else:
        p_conditional_pred.append(p_preds2[T]/(np.ones(q0.shape)@np.linalg.matrix_power(Q, T-1)@q0))

p_conditional = np.zeros(K)
for k in range(K):
    n = num_trial - sum(kc<k)
    for trial in range(num_trial):
        if kc[trial]==k:
            p_conditional[k]+= 1/n

Q_hat = Q.copy()/(lambdaQ_max)
# Q_hat = Q.copy()/np.linalg.norm(Q,1)
Q_hat[np.isnan(Q_hat)] = 0 
test=[q1@p02]
test+= [q1@P2@np.linalg.matrix_power(Q_hat, k)@q0/(sum(q0)) for k in range(K-1)]

plt.figure()
plt.title("Conditional Meeting Chance at K")
plt.plot(np.arange(0,K,1), p_conditional_pred,  "red", linestyle ='--', label="Conditonal First Hitting Model" )
plt.plot(np.arange(0,K,1), p_conditional, '.', color="blue", label="Simulation" )
plt.plot(np.arange(0,K,1), 1/(1+np.linalg.norm(P2-np.linalg.matrix_power(P2,9999)))*np.ones(k+1)*np.inner(p_hat,p_hat), '--', color="blue", label="EST. Lower Bound" )
# plt.plot(np.arange(0,K+1,1), [np.inner(p,p) for p in pk], color="blue", label=r"$\rho_k^T\rho_k$" )
# plt.hlines(np.linalg.norm(B,1),0,K)
plt.plot(test)
plt.legend()
plt.xlabel("K (time step)")
plt.ylabel("Pr(Meet at K given not met before)")




#%% Meeting chance on graph
num_regions = 20
num_robot = 10
num_trial = 1000
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

p = dirichlet.rvs(1*np.ones(num_regions))[0]

S2 = np.sum(p**2)

directed = False

if directed:
    g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
    while not nx.is_strongly_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
else:
    g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)


    
A = nx.adjacency_matrix(g).toarray().T+np.eye(num_regions)

P = FMMC(p, A)
# P = REMC(p, edges)

p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
# p0 = p
K=100
pk = [np.linalg.matrix_power(P , k)@p0 for k in range(K)]
p_meeting = [np.inner(p, p)for p in pk]     


meeting_percentage=np.zeros((num_trial,K))
for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    for k in range(K): 
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                meeting_percentage[trial,k]+= 1/math.comb(num_robot,2)
     
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])    
            

plt.plot(p_meeting)
plt.plot(np.mean(meeting_percentage, axis=0))
plt.hlines(S2,0,K,linestyle='--')
plt.title("meeting chance between any pair")
#%% independent percolation 
num_regions = 20
num_robot = 50
num_trial = 1000
# p = np.random.rand(num_regions)
# p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])

p = dirichlet.rvs(1*np.ones(num_regions))[0]

S2 = np.sum(p**2)

directed = True

if directed:
    g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
    while not nx.is_strongly_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = 2*np.log(num_regions)/num_regions, directed = True)
else:
    g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(num_regions, p = (np.log(num_regions))/num_regions, directed = False)

A = nx.adjacency_matrix(g).toarray().T+np.eye(num_regions)

P = FMMC(p, A)


p0 = dirichlet.rvs(1*np.ones(num_regions))[0]
# p0 = p



#meeting probability prediction
state_comb =  list(product(range(num_regions), repeat=2))
m = len(state_comb)
P2 = np.zeros((m,m))
p02 =np.zeros(m)
for I in  range(m):
    i1,i2 = state_comb[I]
    p02[I] = p0[i1].copy()*p0[i2].copy()
    for J in range(m):
        j1,j2 = state_comb[J]
        P2[I,J] = P[i1,j1].copy()*P[i2,j2].copy()
        
Q = P2.copy()
q0 = p02.copy()

q1 = np.zeros(len(q0))
for I in  range(m):
    i,j = state_comb[I]
    if i==j:
        Q[:,I] = 0#np.nan
        Q[I,:] = 0#np.nan
        q0[I] = 0#np.nan
        q1[I] = 1
K=200   
pk = [np.linalg.matrix_power(P , k)@p0 for k in range(K)]
p_meeting = [np.inner(p, p)for p in pk]     
# absortion time prediction (markov region)
p_first_meeting=[np.inner(p0,p0)]
Q_k = np.eye(m)
P_k = np.eye(m)
for T in np.arange(1,K+1,1):
    P_k = P_k@P2
    p_first_meeting.append(q1.T@P2@Q_k@q0)
    Q_k = Q@Q_k
    
p_meeting_conditional = [p_first_meeting[k]/(1-sum(p_first_meeting[:k])) for k in range(len(p_first_meeting))]
M = np.zeros((K,num_robot,num_robot))
for k in range(K):
    for i in range(num_robot):
        for j in range(num_robot):
            I = j+1 #informed robot
            n = num_robot - I # number of uninformed robot
            r = i - j  #increment of informed robot 
            p_informed = sum([pk[k][ii]*(1-(1-pk[k][ii])**I) for ii in range(num_regions)])
            if i>=j:
                M[k,i,j] = math.comb(n,r)*(1-(1-p_meeting_conditional[k])**I)**r*((1-p_meeting_conditional[k])**I)**(n-r)
                # M[k,i,j] = math.comb(n,r)*(p_informed)**r*(1-p_informed)**(n-r)
        
v = np.zeros(num_robot-1)
v[0]=1

p_percolation_preds=[]
prod = np.eye(num_robot-1)
for T in np.arange(0,K,1):
    prod = M[T,:num_robot-1, :num_robot-1].copy()@prod
    p_percolation_preds.append(1- sum(prod@v))    
k_pred = np.ones(num_robot-1)@np.linalg.inv(np.eye(num_robot-1)-M[-1,:num_robot-1, :num_robot-1])@v

K = int(2.5*k_pred)

kc = np.ones(num_trial)*np.nan
kc_single = np.ones(num_trial)*np.nan

informed_percentage_single_source = np.ones((num_trial, K))
informed_percentage = np.ones((num_trial, K))
r0 = np.random.choice(range(num_regions), 2, p=p0)

for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    r = np.random.choice(range(num_regions), num_robot, p=p0)
    for k in range(K): 
        for i,j in combinations(range(num_robot), 2): 
            if r[i]==r[j]:
                state[i,:] = state[i,:] | state[j,:]
                state[j,:] = state[i,:] | state[j,:]
        informed_percentage_single_source[trial,k] = np.sum(state[:,0])/num_robot
        informed_percentage[trial,k] = np.min(np.sum(state, axis=1)/num_robot)
        
        if (np.prod(state[:,0])):
            kc_single[trial]= np.nanmin([k,kc_single[trial]])
        if (np.prod(state)):
            kc[trial]= k
            break
        for robot in range(num_robot):
            r[robot] = np.random.choice(range(num_regions), 1, p=P[:,r[robot]])    
k_arr = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr.append(k)    
    
k_arr_single = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc_single[i]):]=1
    k_arr_single.append(k)    
#-------------------------------------------------------------------------------------------------        

kc = np.ones(num_trial)*np.nan
kc_single = np.ones(num_trial)*np.nan
informed_percentage = np.ones((num_trial, K))/num_robot
for trial in tqdm(range(num_trial)):
    state = np.eye(num_robot, dtype=bool)
    for k in range(K): 
        for i,j in combinations(range(num_robot), 2): 
            # if np.random.rand()<p_meeting_conditional[k]:
            p_informed = informed_percentage_single_source[trial,k-1]
            if np.random.rand()<p_meeting[k]:
                state[i,:] = state[i,:] | state[j,:]
                state[j,:] = state[i,:] | state[j,:]
        informed_percentage_single_source[trial,k] = np.sum(state[:,0])/num_robot
        informed_percentage[trial,k] = np.min(np.sum(state, axis=1)/num_robot)
        
        if (np.prod(state[:,0])):
            kc_single[trial]= np.nanmin([k,kc_single[trial]])
        if (np.prod(state)):
            kc[trial]= k
            break

k_arr2 = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc[i]):]=1
    k_arr2.append(k)    
    
k_arr_single2 = []
for i in range(num_trial):
    k = np.zeros(K)
    if not np.isnan(kc[i]):
        k[int(kc_single[i]):]=1
    k_arr_single2.append(k)    

plt.figure(dpi=800)
# plt.plot(np.arange(0,K,1), np.mean(k_arr, axis=0), color = "red", label="simulations (multi source)" )
plt.plot(np.arange(0,K,1), np.mean(k_arr_single, axis=0), color = "blue", label="simulations (single source)" )
plt.plot(np.arange(0,K,1), np.mean(k_arr_single2, axis=0), color = "green", label="simulations (meeting chance model)" )

# plt.vlines(k_pred,0,1, label="Mean Passage Time", color="gray", linestyle="--")
plt.plot(np.arange(0,min(K, len(p_percolation_preds)),1), p_percolation_preds[:min(K, len(p_percolation_preds))], color="red", linestyle="--", label = "First Hitting Model")
# plt.plot(np.arange(0,len(p_percolation_preds_fc),1), p_percolation_preds_fc, color="green", linestyle="--", label = "First Hitting Model (FC)")
# plt.plot(np.arange(0,min(K, len(p_percolation_preds)),1), np.array(p_percolation_preds[:min(K, len(p_percolation_preds))])**num_robot, color="orange", linestyle="--", label = "First Hitting Model (Multi-Source)")

plt.xlabel("k (time step)")
plt.ylabel("Full Percolation Probability ")
plt.title(f"Percolation Probability ({num_robot} agents)")


plt.legend()
