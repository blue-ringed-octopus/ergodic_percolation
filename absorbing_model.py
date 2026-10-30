# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:49:07 2026

@author: hibado
"""

import numpy as np
import math 
import tqdm

class Absoorbing_Model:
    def __init__(self,P,p0, num_robot):
        self.P = P
        self.num_robot = num_robot
        self.p0 = p0
        self.pk = [p0]
        self.get_lambda2()
        self.p_not_met=[]
        self.kemeny_factor_k=[]
        self.get_kemeny_factor()
        self.get_kronecker()
        
    def get_kronecker(self):
        P = self.P.copy()
        p0 = self.p0.copy()
        num_regions = len(P)
        
        q_state = []
        idx = []
        for i in range(num_regions):
            for j in range(num_regions):
                if i==j:
                    idx.append(num_regions*i+j)
                else:    
                    q_state.append((i,j))
    
     
        p0_kron = np.kron(p0, p0)
        P_kron= np.kron(P,P)
    
        Q = P_kron.copy()
        Q=np.delete(Q,idx, 0)
        Q=np.delete(Q,idx, 1)
        q0 = p0_kron.copy()
        q0 = np.delete(q0,idx)
        self.kronecker = {"Q": Q, "q":[q0]}
        
    def pred(self, method, K_max=np.inf):
        num_robot = self.num_robot
    
        dist = np.zeros(num_robot-1)
        dist[0]=1
        p_percolation_preds=[]
        prod = np.eye(num_robot-1)   
    
        M = []
        p_percolation = 0
        k = 0
        while p_percolation<=0.9999 and k<K_max:
            M.append(self.Mk(k, method))
            prod = M[-1][:num_robot-1, :num_robot-1].copy()@prod
            p_percolation = 1- sum(prod@dist)
            p_percolation_preds.append(p_percolation)    
            k+=1
        
        dp = np.diff(p_percolation_preds)
        k_pred = np.inner(dp, np.arange(1,k))
        
        return p_percolation_preds, k_pred, np.array(M)
        
    def get_kemeny_factor(self):
        P = self.P.copy()
        P_perp = self.P_perp.copy()
        num_regions = len(P)
        
       
        # P_perp_k = [np.linalg.matrix_power(P_perp,k) for k in range(10000)]
        # Z_k = [np.sum(P_perp_k[:k+1], axis=0) for k in range(9999)]
        Z = np.linalg.inv(np.eye(num_regions)-(P_perp))
        self.Z = Z
        kemeny = np.trace(Z)
        # kemeny_k = np.array([np.trace(x) for x in Z_k])
        self.kemeny_factor = num_regions/kemeny
        # self.kemeny_factor_k = num_regions/kemeny_k
        
    def get_kemeny_factor_k(self,k):
        if len(self.kemeny_factor_k)>k:
            return self.kemeny_factor_k[k]
        
        P_perp = self.P_perp.copy()
        P = self.P.copy()
        num_regions = len(P)
        kemeny_factor_k = self.kemeny_factor_k 

        while len(kemeny_factor_k)<= k:
            Z_k = self.Z@(np.eye(num_regions)-np.linalg.matrix_power(P_perp,k+1))
            kemeny_factor_k.append(num_regions/(np.trace(Z_k)))
        self.kemeny_factor_k = kemeny_factor_k
        
        return self.kemeny_factor_k[k]
    def get_pk(self, k):
        pk=self.pk.copy()
        P = self.P.copy()
        while len(pk)<=k:
            pk.append(P@pk[-1].copy())
        self.pk = pk
        return pk[k]
    
    def get_lambda2(self):
        P = self.P.copy()
        
        lambda_, v = np.linalg.eig(P)
        idx = np.argsort(np.abs(lambda_))
        v1 = v[:,idx[-1]]
        v1 = np.real(v1/np.sum(v1))
        P_infty = np.tile(v1,(len(P),1)).T

        self.P_perp = P-P_infty
        lambda2 = np.abs(lambda_[idx[-2]])
        
        self.v1 = v1
        self.lambda2 = lambda2
        self.lambda2_k = [lambda2**k for k in range(5000)]
        
    def get_p_not_met(self, K):
            
        q = self.kronecker['q'].copy()
        Q = self.kronecker['Q'].copy()
        
        while len(q)<= K:
            q.append(Q@q[-1].copy())
        self.kronecker['q'] = q
        self.p_not_met = np.sum(q, axis=1)
        return self.p_not_met.copy()
    
    def Mk(self,k, method=0):
        num_robot = self.num_robot
        pk = self.get_pk(k)
        lambda2 = self.lambda2
        a = self.lambda2_k.copy()
        kemeny_factor = self.kemeny_factor
        kemeny_factor_k = self.get_kemeny_factor_k(k)
       
        p_not_met = self.get_p_not_met(k)

        M = np.zeros((num_robot,num_robot))
        p_not_meet = (1-(np.inner(pk,pk)))
        for i in range(num_robot):
            for j in range(num_robot):
                I = j+1 #informed robot
                n = num_robot - I # number of uninformed robot
                if n==0:
                    if i>=j:
                        M[i,j] = 1
                else:
                    match method:
                        case 0:
                            #independent
                            p_informed_condition =(1-(p_not_meet)**I)
                        case 1:
                            p_informed_condition = kemeny_factor*(1-(p_not_meet)**I)
                        case 2:
                            p_informed_condition = (kemeny_factor_k)*(1-(p_not_meet)**I)
                        case 3:
                            p_informed_condition = (kemeny_factor)*np.tanh(I*kemeny_factor)*(1-(p_not_meet)**I)
                        case 4:
                            p_informed_condition = (kemeny_factor_k)*np.tanh(I*kemeny_factor_k)*(1-(p_not_meet)**I)
                        case 5:
                            p_informed_condition = (1-(p_not_meet)**(I*(kemeny_factor)))
                        case 6:
                            p_informed_condition = (1-(p_not_meet)**(I*(kemeny_factor_k)))
                        case 7:
                            p_informed_condition = (1-(p_not_meet)**(I*(kemeny_factor*np.tanh(I*(kemeny_factor)))))
                        case 8: 
                            p_informed_condition = (1-(p_not_meet)**(I*(kemeny_factor_k*np.tanh(I*kemeny_factor_k))))
                        case 9:
                            if k==0:
                                p_informed_condition = 1-p_not_met[0]**(I)
                            else:
                                if k==1:
                                    p_not_met_k1 = p_not_met[k-1]**(I)
                                else:
                                    p_not_met_k1 = (1-lambda2**(k-1))*p_not_met[k-1]**(I)+lambda2**(k-1)*p_not_met[k-1]

                                p_not_met_k = (1-lambda2**k)*p_not_met[k]**(I)+lambda2**k*p_not_met[k]
                                p_met_k = 1-p_not_met_k
                                p_met_k1 = 1-p_not_met_k1
                                p_first_meet_k = p_met_k - p_met_k1
                                p_informed_condition = p_first_meet_k/(p_not_met_k1)
                            
                        case 10: 
                            if k==0:
                                p_informed_condition = 1-p_not_met[0]**I
                            else:
                                if k==1:
                                    p_not_met_k1 = p_not_met[k-1]**(I)
                                else:
                                    p_not_met_k1 = p_not_met[k-1]**(I+(1-I)*a[k-1])

                                p_not_met_k = p_not_met[k]**(I+(1-I)*a[k])
                                
                                p_met_k = 1-p_not_met_k
                                p_met_k1 = 1-p_not_met_k1
                                p_first_meet_k = p_met_k - p_met_k1
                                p_informed_condition = p_first_meet_k/(p_not_met_k1)
                                
                        case 11:
                            if k==0:
                                p_informed_condition = 1-p_not_met[0]**(I)
                            else:
                                if k==1:
                                    p_not_met_k1 = p_not_met[k-1]**(I)
                                else:
                                    p_not_met_k1 = p_not_met[k-1]**(I+(1-I)*(kemeny_factor**(k-1)))

                                p_not_met_k = p_not_met[k]**(I+(1-I)*(kemeny_factor**k))
                                
                                p_met_k = 1-p_not_met_k
                                p_met_k1 = 1-p_not_met_k1
                                p_first_meet_k = p_met_k - p_met_k1
                                p_informed_condition = p_first_meet_k/(p_not_met_k1)
                        case 12:
                            if k==0:
                                p_informed_condition = 1-p_not_met[0]**(I)
                            else:
                                p_not_met_k1 = p_not_met[k-1]**(I)
                            
                                p_not_met_k = p_not_met[k]**(I)
                                
                                p_met_k = 1-p_not_met_k
                                p_met_k1 = 1-p_not_met_k1
                                p_first_meet_k = p_met_k - p_met_k1
                                p_informed_condition = kemeny_factor*p_first_meet_k/(p_not_met_k1)
                    if i>=j:
                        r = i - j  #increment of informed robot 
                        M[i,j] = math.comb(n,r)*(p_informed_condition)**r*(1-p_informed_condition)**(n-r)
        return M
 
