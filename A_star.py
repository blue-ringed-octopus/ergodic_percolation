# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:36:56 2024

@author: hibad
"""
import numpy as np

class A_star:
    class Node:
        def __init__(self, id_, coord):
            self.id = id_
            self.coord = coord          
            self.neighbor = []
            self.reset()
            
        def reset(self):
            self.parent=None
            self.g = None
            self.h = None
            self.explored=False
            
    def __init__(self, mask):
        self.grid = mask
        h,w = mask.shape
        id_=0
        nodes=[]
        ids=np.zeros(mask.shape, dtype=np.int32)
        for i in range(w):
            for j in range(h):
                if mask[j,i]:
                    node = self.Node(id_, np.array([i,j]))
                    nodes.append(node)
                    ids[j,i] = id_
                    id_+=1
                else:
                    ids[j,i] = -1
        
        for node in nodes:
            x,y = node.coord
            for i in x+np.array([-1,0,1]):
                if i >0 and i<w:
                    for j in y+np.array([-1,0,1]):
                        if j>0 and j<h: 
                            id_ = ids[j,i]
                            if id_!=-1 and id_!=node.id:
                                node.neighbor.append(nodes[id_])
        self.nodes = nodes  
        self.ids = ids
    
                  
    def search(self, start, target):
        x,y = start
        if (not self.grid[y,x]) or (not self.grid[target[1],target[0]]):
            return False

        if start[0]==target[0] and start[1]==target[1]:
            return []

        x,y = start                                
        current = self.nodes[self.ids[y,x]]
        current.g=0
        current.explored=True
        open_set={current.id: current}
        while len(open_set):
            for node in current.neighbor:
                if not node.explored:
                    node.explored = True
                    node.parent = current
                    node.g = current.g+ np.linalg.norm(node.coord-current.coord)
                    if (node.coord == target).all():
                        return node
                    node.h = np.linalg.norm(node.coord-target)
                    open_set[node.id] = node
                    
                    
                
                if node.g > current.g+np.linalg.norm(node.coord-current.coord):
                    node.parent = current
                    node.g = current.g+np.linalg.norm(node.coord-current.coord)
                                    
            open_set.pop(current.id)
            if len(open_set)==0:
                return False
            
            costs = [node.g+node.h for node in open_set.values()]
            idx = np.argmin(costs)
            current = list(open_set.values())[idx]
            x,y = current.coord
                 
    def plan(self, start,target):
        node = self.search(start, target)
        if not node:
            self.reset()
            return node, None
        
        path=[]
        length = node.g
        while node.parent != None:
            path.append(node.coord)
            node=node.parent
        path.reverse()
        self.reset()
        return path, length
    
    def reset(self):
        for node in self.nodes:
            node.reset()