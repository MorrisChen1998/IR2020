# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:45:28 2019

@author: morri
"""
from sklearn.metrics.pairwise import cosine_similarity

def getSimilarity(docTFIDF, queryTFIDF):
    querysSim=[]
    for query in queryTFIDF:
        sim=[]
        for doc in docTFIDF:
            sim.append(cosine_similarity(query.reshape(1,123), doc.reshape(1,123)))
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))

    return querysSim