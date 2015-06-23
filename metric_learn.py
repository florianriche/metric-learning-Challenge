# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:49:29 2015

@author: richephilippe
"""
import numpy as np

def distances_pairs(X,pairs,dist_func, batch_size=10000):
    n_pairs = pairs.shape[0]
    dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
    for a in range(0, n_pairs, batch_size):
        b = min(a + batch_size, n_pairs)
        dist[a:b] =  [dist_func(X[pairs[i,0],:],X[pairs[i,1],:]) for i in range(a,b)]
    return dist
    
'''
def metric_learnings()'''