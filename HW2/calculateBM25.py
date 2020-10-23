# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:42:43 2019

@author: morri
"""

'''
k1, k3, b, delta, DLN = constant
DLN = document length normalization
'''
import numpy as np

def calculateBM25(k1, k3, b, delta, DLN, TFq, TFj, IDF):
    TFpron = np.array(TFj)/(1-b+b*DLN)+delta
    F = (k1+1)*TFpron/(k1+TFpron)
    TF = (k3+1)*np.array(TFq)/(k3+np.array(TFq))
    SIMbm25 = np.sum(F*TF*np.array(IDF))
    return SIMbm25
  

def getSimilarity(k1, k3, b, delta, DLNs, queryTF, docTF, IDF):
    querysSim=[]
    for query in queryTF:
        sim=[]
        for doc in docTF:
            sim.append(calculateBM25(k1, k3, b, delta, DLNs[len(sim)], query, doc, IDF))
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))

    return querysSim
