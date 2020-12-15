# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:45:28 2019

@author: morri
"""
import operator
import numpy as np
from numba import jit
from tqdm import tqdm

#%%
@jit(nopython=True)#with numba to accelerate
def calculateCosineValue(q,d):
    dot = np.dot(q, d)
    q_distance = np.linalg.norm(q)
    d_distance = np.linalg.norm(d)
    if(q_distance==0 or d_distance==0):
        return 0
    else:
        return dot / (q_distance*d_distance)
#%%
def getSimilarity(docTF,queryIDs,querysSim,idf,topK,a,b):
    new_querysSim=[]
    for q in tqdm(range(len(querysSim))):
        sim={}
        sum_Reldoc=np.zeros(len(idf))
        for i in range(topK):
            sum_Reldoc = sum_Reldoc + np.squeeze(docTF[querysSim[q][i]].toarray())*(topK-i)*b
        sum_Reldoc/=(topK*sum(list(range(topK+1))))
        for w in range(len(queryIDs[q])):
            sum_Reldoc[queryIDs[q][w]] += a/len(queryIDs[q])
        
        for j in range(len(querysSim[q])):
            sim[querysSim[q][j]]=calculateCosineValue(sum_Reldoc, np.squeeze(docTF[querysSim[q][j]].toarray()))
        new_querysSim.append(sorted(sim.items(), key=operator.itemgetter(1),reverse=True))

    return new_querysSim