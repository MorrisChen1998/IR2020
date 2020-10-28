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

def calculateBM25(k1, k3, b, delta, DLN, TFq, TFd, IDF):
    TFpron = TFd/(1-b+b*DLN)+delta
    F = (k1+1)*TFpron/(k1+TFpron)
#    F = (k1+1)*TFd/(k1*(1-b+b*DLN)+TFd)+delta
#    TF = (k3+1)*TFq/(k3+TFq)
    SIMbm25 = np.sum(F*TFq*IDF*IDF)
    return SIMbm25

def getSimilarity(k1, k3, b, delta, DLNs, queryTF, docTF, IDF):
    querysSim=[]
    for query in queryTF:
        sim=[]
        for doc in docTF:
            sim.append(calculateBM25(k1, k3, b, delta, DLNs[len(sim)], query, doc, IDF))
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))
    return querysSim
