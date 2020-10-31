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

def calculateBM25L(k1, k3, b, delta, DLN, TFq, TFd, IDF):
    TFpron = TFd/(1-b+b*DLN)+delta
    F = (k1+1)*TFpron/(k1+TFpron)
    TF = (k3+1)*TFq/(k3+TFq)
    SIMbm25L = np.sum(F*TF*IDF*IDF)
    return SIMbm25L
def calculateBM25(k1, k3, b, delta, DLN, TFq, TFd, IDF):
    F = (k1+1)*TFd/(k1*(1-b+b*DLN)+TFd)
    TF = (k3+1)*TFq/(k3+TFq)
    SIMbm25 = np.sum(F*TF*IDF*IDF)
    return SIMbm25
def calculateBM1(TFq,IDF):
    return np.dot(IDF,TFq)

def getSimilarity(a1, a2, k1, k3, b, delta, DLNs, queryTF, docTF, IDF):
    querysSim=[]
    for query in queryTF:
        sim=[]
        for doc in docTF:
            BM25L = calculateBM25L(k1, k3, b, delta, DLNs[len(sim)], query, doc, IDF)
            BM25 = calculateBM25(k1, k3, b, delta, DLNs[len(sim)], query, doc, IDF)
            BM1 = calculateBM1(query,IDF)
            sim.append(a1*BM25L+a2*BM25+(1-a1-a2)*BM1)
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))
    return querysSim
