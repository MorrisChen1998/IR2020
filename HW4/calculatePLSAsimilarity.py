# -*- coding: utf-8 -*-

import numpy as np

def calculatePLSA(a, b, dictionary, queryID, docTF, P_w_dj, BG, P_T_dj, P_w_T):
    P_q_d = 0
    for word in range(len(queryID)):
        unigram_part = np.log(P_w_dj[word]*a) if P_w_dj[word]>0 else 0
        
        sigma = np.dot(P_w_T[:,queryID[word]],P_T_dj)
        topic_part = np.log(sigma*b) if sigma>0 else 0
        
        bg_part = np.log(BG[queryID[word]]*(1-a-b)) if BG[queryID[word]]>0 else 0
        
        P_q_d += np.log(np.exp(unigram_part) + np.exp(topic_part) + np.exp(bg_part))
    return P_q_d

def getSimilarity(a, b, dictionary, queryIDs, docTF, P_w_d, BG, P_T_d, P_w_T):
    querysSim=[]
    for query in range(len(queryIDs)):
        sim=[]
        for doc in range(len(docTF)):
            sim.append(calculatePLSA(a, b, dictionary, queryIDs[query], docTF[doc], P_w_d[doc], BG, P_T_d[doc], P_w_T))
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True)[:1000])
    return querysSim
