# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:45:28 2019

@author: morri
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculateCosineValue(q,d):
    dot = np.dot(q, d)
    q_distance = np.linalg.norm(q)
    d_distance = np.linalg.norm(d)
    if(q_distance==0 or d_distance==0):
        return 0
    else:
        return dot / (q_distance*d_distance)

def getSimilarity(docTFIDF, queryTFIDF):
    querysSim=[]
    for query in queryTFIDF:
        sim=[]
        for doc in docTFIDF:
#            sim.append(cosine_similarity(query.reshape(1,123), doc.reshape(1,123)))
            sim.append(calculateCosineValue(query, doc))
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))

    return querysSim