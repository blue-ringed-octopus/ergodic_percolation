#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:53:55 2024

@author: hibad
"""
import yaml
import pickle
import cv2 
import numpy as np
from hierarchical_graph import Hierarchical_Graph, Graph

class Map_Manager:
    def __init__(self,n=7):
        scale =  4
        map_ = cv2.imread("resources/circle_map.png")
  

        ring = np.where((map_[:,:,2]>0) * (map_[:,:,0]==0) * (map_[:,:,1]==0))
        map_grey =  cv2.cvtColor(map_, cv2.COLOR_BGR2GRAY)
        map_grey[ring] = 0
        region_map = np.zeros(map_grey.shape, np.int32)
        region_idx ={}
        region_map[np.where(map_grey== 0)] = -1
        ring_idx = {}
        for i in range(n):
            idx = np.where(map_grey== 30*(i+1))
            region_map[idx] = i
            region_idx[i]=np.vstack(idx).T
            ring_idx[i] = np.where(map_[:,:,2]== 30*(i+1) * (map_[:,:,0]==0) * (map_[:,:,1]==0))
            
        self.n = n   
        self.masked_ring_map = np.ma.masked_where((region_map>=0) + (map_[:,:,2]>0),region_map)
        self.masked_region_map = np.ma.masked_where(region_map>=0,region_map)
        self.build_region_graph()
        self.ring_idx = ring_idx
        
    def build_region_graph(self):
    
        root_node = Hierarchical_Graph.Node(0,[0,0], 0)
        root_grid = self.masked_region_map.data.copy()
        root_grid[self.masked_region_map.mask] = 0
        root_graph = Graph({0:root_node} , 0 ,root_grid)
        h_graph = Hierarchical_Graph(root_graph)
        
           
        region_nodes={}
        region_idx={}
        idx_map = self.masked_region_map.data.copy()
        for i in range(self.n):
           idx= np.array(np.where(idx_map==i)).T
           region_idx[i] = idx
           region_nodes[i] = Hierarchical_Graph.Node(i,np.mean(idx,0), 1)
           region_nodes[i].add_parent(root_node)
        
        region_graph =  Graph(region_nodes , 1 ,idx_map)
        h_graph.levels[1] = region_graph
        
        stencil = [[i,j] for i in [-1,0,1] for j in [-1,0,1] if not(i==0 and j==0)]
        
        grid_nodes, grid_ids = h_graph.grid2graph(stencil, 2)
        grid_graph = Graph(grid_nodes , 2 ,grid_ids)
        h_graph.levels[2] = grid_graph
        self.hierarchical_graph = h_graph
        # self.region_map = region_map
        
    def get_index(self, coord):
        idx = (coord[0:2]-self.costmap["bounds"]["min"][0:2])/self.costmap["resolution"]
        idx = np.array([round(idx[0]), round(idx[1])])
        return idx
    
    def get_region_graph_img(self):
        return self.hierarchical_graph.levels[1].plot_graph()
    
    
    def get_graph(self, level):
        ids = list(self.hierarchical_graph.levels[level].nodes.keys())
        edges = self.hierarchical_graph.get_edges(level)
        # location = np.array([ node.coord for node in self.hierarchical_graph.levels[level].nodes.values()])
        return ids,  edges
    
    # def coord_to_region(self, coord, level):
    #     idx = self.get_index(coord)
    #     # region = self.hierarchical_graph.levels[level].id_map[idx[0], idx[1]]
    #     region = self.region_map[idx[0], idx[1]]
    #     return int(region)
    
    
if __name__ == '__main__':
#     import matplotlib.pyplot as plt
    manager = Map_Manager()
#     with open('tests/detections.pickle', 'rb') as f:
#         dat = pickle.load(f)
        
#     pc=o3d.geometry.PointCloud()
#     pc.points=o3d.utility.Vector3dVector(dat["cloud"])
#     manager.process_reference(pc)
#     manager.set_entropy(dat["p"][-1], np.array(range(len(dat["p"][-1]))))
#     test = manager.visualize_entropy()
#     # o3d.visualization.draw_geometries([test])
#     print(manager.get_entropy())
#     img = manager.get_region_graph_img()
#     plt.figure(dpi=1200)   
#     plt.imshow(img)    
#     ids, edges , h = manager.get_graph(1)
#     region = manager.coord_to_region([2.57,-0.75], 1)
#     img2 = manager.draw_graph_entropy()
#     plt.figure()   
#     plt.imshow(img2)    


   