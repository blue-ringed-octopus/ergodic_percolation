# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 18:40:22 2025

@author: hibado
"""

import numpy as np
import matplotlib.pyplot as plt 

ptp = 0.06
n=100

p = ptp
ps = [p]

for k in np.arange(2,50,1):   
    p_noindirect = 1
    for pk in ps:
        p_noindirect *= (1-ptp*(pk))
    p = 1-((1-ptp)**k)*(p_noindirect)**((n-2))
    ps.append(p)

 
ps = np.array(ps)
plt.figure()    
plt.plot(np.arange(1,50,1),ps**(n**2-n), ".")
plt.xlabel("k")
plt.ylabel(r"$p^n$")
#%%
def p_connected(ptp, k, n):
    if n == 2:
        return 1-(1-ptp)**k
    
    thres = 0.5
    if ptp==0:
        return np.inf
    if ptp>=thres:
        return 1
    ps=[ptp]
    k=1
    while ps[-1]<thres:
        k+=1
        p_noindirect = 1
        for pk in ps:
            p_noindirect *= (1-ptp*(pk))
        p = 1-((1-ptp)**k)*(p_noindirect)**((n-2))
        ps.append(p)

    return (k-1)+1/(ps[-1]-ps[-2])*(thres-ps[-2])

n=100
thres = 0.5
kc=[]
num_points = 5000
for ptp in np.linspace(0.01,thres-0.01,num_points):
    p = ptp
    ps = [p]
    k=1
    while p<thres:   
        k+=1
        p_noindirect = 1
        for pk in ps:
            p_noindirect *= (1-ptp*(pk))
        p = 1-((1-ptp)**k)*(p_noindirect)**((n-2))
        ps.append(p)
    kc.append((k-1)+1/(ps[-1]-ps[-2])*(thres-ps[-2]))

plt.figure()    
plt.plot(np.linspace(0.06,1,num_points), kc, ".")
plt.xlabel("pTp")
plt.ylabel(r"$K_c$")

#%%
ptp = 0.01
kc=[]
thres = 0.5
for n in np.arange(2,250,1):
    p = ptp
    ps = [p]
    k=1
    while p<thres: 
        k+=1
        p_noindirect = 1
        for pk in ps:
            p_noindirect *= (1-ptp*(pk))
        p = 1-((1-ptp)**k)*(p_noindirect)**((n-2))
        ps.append(p)
    kc.append((k-1)+1/(ps[-1]-ps[-2])*(thres-ps[-2]))
plt.figure()        
plt.plot(np.arange(2,250,1), kc, ".")
plt.xlabel("n")
plt.ylabel(r"$K_c$")