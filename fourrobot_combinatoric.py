# -*- coding: utf-8 -*-
"""
Created on Wed Dec 01 22:47:26 0225

@author: hibado
"""
import numpy as np
from copy import deepcopy

# source0=[["(01)","(02)"],
#         ["(12)<(02)"],
#         ["(12)<(01)"]
#          ]
# source1=[["(01)","(12)"],
#         ["(02)<(12)"],
#         ["(02)<(01)"]
#          ]
# source2=[["(12)","(02)"],
#         ["(01)<(02)"],
#         ["(01)<(12)"]
#          ]
#%%
source0=[["(01)","(02)","(03)"],
         ["(12)<(13)<(03)"],
         ["(12)<(23)<(03)"],
         ["(13)<(23)<(02)"],
         ["(13)<(12)<(02)"],
         ["(23)<(13)<(01)"],
         ["(23)<(12)<(01)"],
         ["(23)<(03)","(01)"],
         ["(23)<(02)","(01)"],
         ["(13)<(01)","(02)"],
         ["(13)<(03)","(02)"],
         ["(12)<(02)","(03)"],
         ["(12)<(01)","(03)"],
         ["(12)<(02)","(23)<(02)"],
         ["(13)<(03)","(23)<(03)"],
         ["(12)<(01)","(13)<(01)"],
         ]

source1=[["(01)","(12)","(13)"],
         ["(02)<(03)<(13)"],
         ["(02)<(23)<(13)"],
         ["(03)<(23)<(12)"],
         ["(03)<(02)<(12)"],
         ["(23)<(03)<(01)"],
         ["(23)<(02)<(01)"],
         ["(23)<(13)","(01)"],
         ["(23)<(12)","(01)"],
         ["(03)<(01)","(12)"],
         ["(03)<(13)","(12)"],
         ["(02)<(12)","(13)"],
         ["(02)<(01)","(13)"],
         ["(02)<(12)","(23)<(12)"],
         ["(03)<(13)","(23)<(13)"],
         ["(02)<(01)","(03)<(01)"],
         ]

source2=[["(12)","(02)","(23)"],
         ["(01)<(13)<(23)"],
         ["(01)<(03)<(23)"],
         ["(13)<(03)<(02)"],
         ["(13)<(01)<(02)"],
         ["(03)<(13)<(12)"],
         ["(03)<(01)<(12)"],
         ["(03)<(23)","(12)"],
         ["(03)<(02)","(12)"],
         ["(13)<(12)","(02)"],
         ["(13)<(23)","(02)"],
         ["(01)<(02)","(23)"],
         ["(01)<(12)","(23)"],
         ["(01)<(02)","(03)<(02)"],
         ["(13)<(23)","(03)<(23)"],
         ["(01)<(12)","(13)<(12)"],
         ]
source3=[["(13)","(23)","(03)"],
         ["(12)<(01)<(03)"],
         ["(12)<(02)<(03)"],
         ["(01)<(02)<(23)"],
         ["(01)<(12)<(23)"],
         ["(02)<(01)<(13)"],
         ["(02)<(12)<(13)"],
         ["(02)<(03)","(13)"],
         ["(02)<(23)","(13)"],
         ["(01)<(13)","(23)"],
         ["(01)<(03)","(23)"],
         ["(12)<(23)","(03)"],
         ["(12)<(13)","(03)"],
         ["(12)<(23)","(02)<(23)"],
         ["(01)<(03)","(02)<(03)"],
         ["(12)<(13)","(01)<(13)"],
         ]

def combine(s1,s2):
    result = []
    for i in s1:
        for j in s2:
            item=i+j
           
            result.append(item)

    complete = result[0]
    poped=[]
    for i,ii in enumerate(result):
        state = np.zeros(len(complete), dtype=bool)
        for item in ii:
            for j,jj in enumerate(complete):
                    state[j] += jj in item
                
        if state.all() and not i==0:
            poped.append(result.pop(i))
            
    for ii, condition in enumerate(result):
        remove=[]
        for i in range(len(condition)):
            for j in range(len(condition)):
                if not (i==j or j in remove):
                    if condition[i] in condition[j]:
                        remove.append(i)
        result[ii]=[ condition[i] for i in range(len(condition)) if i not in remove]
    
    remove=[]
    for i in range(len(result)):
        if i not in remove:
            condition = result[i]
            for j in range(len(result)):
                state=np.zeros(len(condition), dtype=bool)
                if not i== j:
                    for ii, item in enumerate(condition):
                        state[ii] = item in result[j]
                if state.all():
                    remove.append(j)
                    
    result=[result[i] for i in range(len(result)) if i not in remove]
    return result

result = combine(source0,source1)
result = combine(result,source2)
result = combine(result,source3)
remove=[]
for k, condition in enumerate(result):
    if k not in remove:
        for i,ii in enumerate(result):
            if not i==k:
                state = np.zeros(len(condition), dtype=bool)
                for item in ii:
                    for j,jj in enumerate(condition):
                            state[j] += jj in item
                        
                if state.all():
                    remove.append(i)
                
result=[result[i] for i in range(len(result)) if i not in remove]
                

