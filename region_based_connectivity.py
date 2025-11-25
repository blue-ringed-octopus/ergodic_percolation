# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:17:30 2025

@author: hibado
"""


import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt 
import pickle 
num_robot = 100
num_trials = 100
num_regions = 100
K = 50
critical=np.zeros(num_trials)*np.nan
p_arr = []
for trial in range(num_trials):
    G=nx.DiGraph()
    p = np.random.rand(num_regions)
    p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
    reachable = np.eye(num_robot)

    p_arr.append(p)
    for k in range(K):
        r = np.random.choice(range(num_regions), num_robot, p=p)
        for i in range(num_robot):
            G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k+1)+"]")
            G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k)+"]")

        for i,j in combinations(range(num_robot), 2): 
            if r[i] == r[j]:
                G.add_edge(str(i)+"["+str(k)+"]", str(j)+"["+str(k)+"]")
                G.add_edge(str(j)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
        for i in range(num_robot):
            for j in range(num_robot):
                if not reachable[i,j]:
                    reachable[i,j] = nx.has_path(G,str(i)+"["+str(0)+"]", str(j)+"["+str(k)+"]")
        if reachable.all():
            critical[trial]=k
            break
#%%
pc =  np.log(num_robot)/num_robot
pTp = np.linspace(1/num_regions,1,100)
ln_p_arr = [-np.log(np.inner(p,p)) for p in p_arr]
m, _, _, _ = np.linalg.lstsq(np.array([ln_p_arr]).T,  critical, rcond=None)
test = 1-2*np.log(pc)
plt.figure()
plt.plot(ln_p_arr ,critical, '.', color="r", alpha=0.2)
#plt.plot([np.var(p) for p in p_arr], critical, '.', color="r", alpha=0.5)
plt.plot(-np.log(pTp) , m*(-np.log(pTp)), '--',color="r")
# plt.plot(pTp , -1/np.log(1-pTp), '--',color="r")
plt.xlabel(r"$-ln(p^Tp)$")
plt.ylabel(r"$K_c$")
plt.yticks(np.arange(0,int(np.nanmax(critical)),3))
plt.title("Kc vs -ln(pTp)")
plt.legend()

plt.figure()
plt.plot([np.inner(p,p) for p in p_arr], critical, '.', color="r", alpha=0.2)
plt.plot(pTp , m*(-np.log(pTp))**0.5, '--',color="r")
plt.vlines(0,np.nanmax(critical), 1/num_regions, linestyle='--')
plt.xlabel(r"$p^Tp$")
plt.ylabel(r"$K_c$")
plt.yticks(np.arange(0,int(np.nanmax(critical)),3))
plt.title("Kc vs pTp")
plt.legend()

b,c2 = np.polyfit(np.log(ln_p_arr), np.log(critical+1), 1)
c = np.exp(c2/b)
plt.figure()
plt.plot(np.log(ln_p_arr) ,np.log(critical+1), '.', color="r", alpha=0.2)
plt.plot(np.log(-np.log(pTp)) ,b*np.log(-np.log(pTp))+c2, '--', color="r")
plt.xlabel(r"$ln(-ln(p^Tp))$")
plt.ylabel(r"$ln(K_c)$")
plt.legend()

plt.figure()
plt.plot([np.inner(p,p) for p in p_arr], critical+1, '.', color="r", alpha=0.2)
plt.plot(pTp , (-c*np.log(pTp))**b, '--',color="r")
plt.vlines(0,np.nanmax(critical), 1/num_regions, linestyle='--')
plt.xlabel(r"$p^Tp$")
plt.ylabel(r"$K_c$")
plt.yticks(np.arange(0,int(np.nanmax(critical)),3))
plt.title("Kc vs pTp")
plt.legend()

plt.figure()
plt.plot([np.inner(p,p) for p in p_arr], critical, '.', color="r", alpha=0.2)
plt.plot(pTp , (-num_robot*np.log(pTp))**0.5, '--',color="r")
plt.vlines(0,np.nanmax(critical), 1/num_regions, linestyle='--', label="uniform")
plt.xlabel(r"$p^Tp$")
plt.ylabel(r"$K_c$")
plt.yticks(np.arange(0,int(np.nanmax(critical)),3))
plt.title("Kc vs pTp")
plt.legend()
#%%
K = 50
num_trials = 30
num_regions = 100
p = np.random.rand(num_regions)
p[1:] = p[1:]/np.sum(p[1:])*(1-p[0])
pTp = np.inner(p,p)
print(pTp)
#%%
num_robots = np.arange(2,250, 10)
critical=np.zeros(((len(num_robots)),num_trials))*np.nan
for ii, num_robot in enumerate(num_robots):
    print(str(ii)+"/"+ str(len(num_robots)))
    for trial in range(num_trials):
        G=nx.DiGraph()
        reachable = np.eye(num_robot)
        for k in range(K):
            r = np.random.choice(range(num_regions), num_robot, p=p)
            for i in range(num_robot):
                G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k+1)+"]")
                G.add_edge(str(i)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
    
            for i,j in combinations(range(num_robot), 2): 
                if r[i] == r[j]:
                    G.add_edge(str(i)+"["+str(k)+"]", str(j)+"["+str(k)+"]")
                    G.add_edge(str(j)+"["+str(k)+"]", str(i)+"["+str(k)+"]")
            for i in range(num_robot):
                for j in range(num_robot):
                    if not reachable[i,j]:
                        reachable[i,j] = nx.has_path(G,str(i)+"["+str(0)+"]", str(j)+"["+str(k)+"]")
                            
            if reachable.all():
                critical[ii,trial]=k
                break
  #%%
x =  np.linspace(2,250, 1000)
# plt.plot(x,np.exp(2.55)*(np.log(x)**2/(x)**0.81)*(-np.log(pTp))**0.5, '--', color="r")
# plt.plot(x,np.exp(3.35)*(np.log(0.2*x+0.8)**2/(x)**0.83)*(-np.log(pTp))**1, '--', color="r")
plt.plot(x,np.exp(0)*(np.log(0.3*x+0.7)**2/(x)**((1-pTp)))*(-np.log(pTp))**1, '--', color="r")

plt.plot(2,1/pTp-1, 'x', color="r")

plt.errorbar(num_robots, np.mean(critical, axis=1), np.std(critical, axis=1)/np.sqrt(num_trials), fmt='.')
plt.plot(num_robots, np.quantile(critical, 0.75, axis=1), '.', color="b", alpha=0.1)
plt.plot(num_robots, np.quantile(critical, 0.25, axis=1), '.', color="b", alpha=0.1)
plt.xlim([0,250])
plt.xlabel(r"$N$")
plt.ylabel(r"$K_c$")
# plt.yticks(np.arange(0,int(np.nanmax(np.quantile(critical, 0.75, axis=1))),3))
plt.title("Kc vs number of robots, pTp = " + str(pTp))
plt.legend()
#%%
ln_n = np.linspace(np.log(2),6,1000)
a=2 
b=0.81
c = np.log(0.31)
plt.figure()
#plt.plot(ln_n,2*np.log(ln_n)-0.81*ln_n+0.5*np.log((-np.log(pTp)))+2.55, '--', color="r")
plt.plot(ln_n,a*np.log(ln_n+c)-b*ln_n+np.log((-np.log(pTp)))+3, '--', color="r")
plt.errorbar(np.log(num_robots), np.mean(np.log(critical), axis=1), np.std(np.log(critical), axis=1)/np.sqrt(num_trials), fmt='.')
plt.plot(np.log(2),np.log(1/pTp), 'x', color="r")


# plt.xlim([0,250])
# plt.ylim([0,2.5])

plt.xlabel(r"$N$")
plt.ylabel(r"$K_c$")
plt.title("Kc vs number of robots, pTp = " + str(pTp))
#%%
dat = {"p":p, "K": critical,"N": num_robots }
with open('data/3.p', 'wb') as file:
     pickle.dump(dat, file)