# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:42:43 2019

@author: morri
"""

'''
k1, k3, b, delta, DLN = constant
DLN = document length normalization
'''
import operator
import numpy as np
from numba import jit
from tqdm import tqdm

#%%
@jit(nopython=True)
def calculateBM25L(k1, k3, b, delta, DLN, TFq, TFd, IDF):
    TFpron = TFd/(1-b+b*DLN)+delta
    F = (k1+1)*TFpron/(k1+TFpron)
    TF = (k3+1)*TFq/(k3+TFq)
    SIMbm25L = np.sum(F*TF*IDF*IDF)
    return SIMbm25L
@jit(nopython=True)
def calculateBM25(k1, k3, b, delta, DLN, TFq, TFd, IDF):
    F = (k1+1)*TFd/(k1*(1-b+b*DLN)+TFd)
    TF = (k3+1)*TFq/(k3+TFq)
    SIMbm25 = np.sum(F*TF*IDF*IDF)
    return SIMbm25
@jit(nopython=True)
def calculateBM1(TFq,IDF):
    return np.dot(IDF,TFq)
#%%
def getSimilarity(a1, a2, k1, k3, b, delta, DLNs, docTF,queryIDs,querysSim,idf,topK,alpha,beta):
    new_querysSim=[]
    for q in tqdm(range(len(querysSim))):
        sim=np.zeros(len(querysSim[q]))
        sum_Reldoc=np.zeros(len(idf))
        for i in range(topK):
            sum_Reldoc = sum_Reldoc + np.squeeze(docTF[querysSim[q][i]].toarray())*idf*(topK-i)*beta
        sum_Reldoc/=(topK*sum(list(range(topK+1))))
        for w in range(len(queryIDs[q])):
            sum_Reldoc[queryIDs[q][w]] += alpha/len(queryIDs[q])
        
        for j in range(len(querysSim[q])):
            BM25L = calculateBM25L(k1, k3, b, delta, DLNs[querysSim[q][j]], sum_Reldoc, np.squeeze(docTF[querysSim[q][j]].toarray()), idf)
            # BM25 = calculateBM25(k1, k3, b, delta, DLNs[querysSim[q][j]], sum_Reldoc, np.squeeze(docTF[querysSim[q][j]].toarray()), idf)
            # BM1 = calculateBM1(sum_Reldoc,idf)
            # sim[querysSim[q][j]]=a1*BM25L+a2*BM25+(1-a1-a2)*BM1
            sim[querysSim[q][j]]=BM25L
        new_querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))
    return new_querysSim
