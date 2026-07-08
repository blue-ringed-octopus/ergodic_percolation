# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:52:53 2024

@author: hibad
"""
import numpy as np
import cvxpy as cp
import cv2
np.set_printoptions(precision=2)

def MH(rho_bar, edges):
    n = len(rho_bar)
    G = np.zeros((n, n))
    for i,j in edges:
        if not i == j :
            G[j,i] = 1
    G=G/np.sum(G, 0) 
    P = np.zeros((n,n))
    for i,j in edges:
        if not i==j: 
            P[j,i] = G[j,i]* min((rho_bar[j]*G[i,j])/(rho_bar[i]* G[j,i]), 1 )
    
    for i in range(n):
        P[i,i] = 1 - np.sum(P[:,i])
        
    return P

def FB_REMC(rho_hat_k, rho_bar, k, rho_k, edges):
    n = len(rho_bar)

    P= cp.Variable((n,n))
    b = (k+1)*rho_bar - rho_hat_k*k
    objective = cp.norm(P@rho_k - b)
    constrains = [np.ones(n)@P == np.ones(n),
                  cp.min(P)>= 0,
            #      P@rho_bar == rho_bar
                  ]
    
    for i in range(n):
        for j in range(n):
            if ([i,j] not in edges):
                constrains.append(P[j,i]==0)
    prob = cp.Problem(cp.Minimize(objective),
                           constrains)
    prob.solve()
    P=P.value
    P[P<0]=0
    for i in range(n):
        for j in range(n):
            if ([i,j] not in edges):
                P[j,i]=0
    P=P/np.sum(P, 0) 
    return P

def REMC(weight, edges):
    n = len(weight)
    w=weight/sum(weight)

    w = w + 0.001
    w = w/sum(w)

    P= cp.Variable((n,n))
    
    
    q = np.sqrt(w)
    Q=np.diag(q)

    constrains=[
                P@w==w,
                cp.min(P)>=0,
                np.ones(n)@P==np.ones(n)
                ]
    
    for i in range(n):
        for j in range(n):
            if ((i,j) not in edges):
                constrains.append(P[j,i]==0)
    # P_tilde = P - 2*outer(w,np.ones(n))
    P_tilde = np.linalg.inv(Q)@P@(Q)-2*cp.outer(q,q)
    

    objective = cp.lambda_max(1/2*(P_tilde+P_tilde.T))

    prob = cp.Problem(cp.Minimize(objective),
                      constrains)

    prob.solve()
    P=P.value
    
    P[P<0]=0
    for i in range(n):
        for j in range(n):
            if ((i,j) not in edges):
                P[j,i]=0
    P=P/sum(P, 0) 
    return P

def FMMC(weight, A, transform=True, reversible = False):
    n = len(weight)
    w=weight/sum(weight)
    P= cp.Variable((n,n))


    q=np.sqrt(w)
    Q=np.diag(q)
    constrains=[
                cp.min(P)>=0,
                np.ones(n)@P==np.ones(n)]
    if reversible:
        constrains.append(P@np.diag(w) == np.diag(w)@P.T)
    else:
        constrains.append(P@w==w)
        
    for i in range(n):
        for j in range(n):
            if not A[j,i]:
                constrains.append(P[j,i]==0)
               
    if transform:
        prob = cp.Problem(cp.Minimize(cp.norm(np.linalg.inv(Q)@P@(Q)-cp.outer(q,q),2)),
                            constrains)

    else:
          prob = cp.Problem(cp.Minimize(cp.norm(P-cp.outer(w,np.ones(n)),2)),
                            constrains)
 

    prob.solve()
    P=P.value

    P[P<0]=0
    for i in range(n):
        for j in range(n):
            if not A[j,i]:
                P[j,i]=0
    P=P/sum(P, 0)

    return P
# class Graph_Planner:
#     def __init__(self, nodes, edges, strategy = "ergodic"):
#         self.strategy = strategy 
#         self.num_regions = len(nodes)
#         self.edges = edges
#         self.adjacency = np.zeros((self.num_regions,self.num_regions))
#         for i, j in self.edges:
#             self.adjacency[j,i]=1
        
#     def FB_REMC(self, rho_hat_k, rho_bar, k, rho_k):
#         n = self.num_regions
#         P= cp.Variable((n,n))
#         b = (k+1)*rho_bar - rho_hat_k
#         objective = cp.norm(P@rho_k - b)
#         constrains = [np.ones(n)@P == np.ones(n),
#                       cp.min(P)>= 0,
#                    #   P@rho_bar == rho_bar
#                       ]
        
#         for i in range(n):
#             for j in range(n):
#                 if ([i,j] not in self.edges):
#                     constrains.append(P[j,i]==0)
#         prob = cp.Problem(cp.Minimize(objective),
#                                constrains)
#         prob.solve()
#         P=P.value
#         P[P<0]=0
#         for i in range(n):
#             for j in range(n):
#                 if ([i,j] not in self.edges):
#                     P[j,i]=0
#         P=P/sum(P, 0) 
#         return P  
   
#     def FMMC(self, weight, transform=True):
#         w =weight/sum(weight)
#         n = self.num_regions
#         P= cp.Variable((n,n))


#         q=np.sqrt(w)
#         Q=np.diag(q)
#         constrains=[
#                    # P@w==w,
#                     P@np.diag(w) == np.diag(w)@P.T,
#                     cp.min(P)>=0,
#                     np.ones(n)@P==np.ones(n)]
        
#         for i in range(n):
#             for j in range(n):
#                 if ([i,j] not in self.edges):
#                     constrains.append(P[j,i]==0)

#         if transform:
#             prob = cp.Problem(cp.Minimize(cp.norm(np.linalg.inv(Q)@P@(Q)-cp.outer(q,q),2)),
#                                 constrains)

#         else:
#               prob = cp.Problem(cp.Minimize(cp.norm(P-cp.outer(w,np.ones(n)),2)),
#                                 constrains)
     

#         prob.solve()
#         P=P.value

#         P[P<0]=0
#         for i in range(n):
#             for j in range(n):
#                 if ([i,j] not in self.edges):
#                     P[j,i]=0
#         P=P/sum(P, 0)

#         return P
#     def set_weights(self, weight):
#         self.w = weight.copy()
#         if self.strategy == "ergodic":
#             P = REMC(weight, self.edges )
            
#         elif self.strategy == "random":
#             P = self.adjacency.copy()
#             P=P/sum(P, 0)
            
#         elif self.strategy == "greedy":
#             print(weight/np.sum(weight))
#             P =  self.adjacency.copy()
#             for i in range(self.num_regions):
#                 P[:, i] = P[:, i] * weight 
#                 P[:, i]= P[:, i]==np.max(P[:, i])
#         self.P= P
#         return P.copy()
    
#     def get_next_region(self, current_region):
#         P = self.P.copy()
#         region = np.random.choice(range(self.num_regions),p=P[:,current_region])
#         return region, P
    



if __name__ == '__main__':
    pass