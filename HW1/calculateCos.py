# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:45:28 2019

@author: morri
"""
import operator
import numpy as np

def calculateCosineValue(q,dj):#parameters are vectors which present query's and doc's TFIDF
     return np.dot(q, dj) / (np.linalg.norm(q)*np.linalg.norm(dj))
        

def getSimilarity(docTFIDF, queryTFIDF):
    querysSim=[]
    for query in queryTFIDF:
        sim=[]
        for doc in docTFIDF:
            sim.append(calculateCosineValue(query, doc))
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))

    return querysSim