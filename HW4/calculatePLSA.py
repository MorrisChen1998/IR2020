# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

def calculateP_q_d(a, b, query_len, P_wi_dj, bg_i, P_T_dj, P_wi_T):
    P_q_d = 0
    for i in range(query_len):
        # unigram_model = np.log(a)+np.log(P_wi_dj[i] if P_wi_dj[i]>1e-50 else 1e-50)
        topic_model = np.log(b)+np.log(np.dot(P_T_dj,P_wi_T[i]))
        bg_model = np.log(1-a-b)+np.log(bg_i[i] if bg_i[i]>1e-50 else 1e-50)
        P_q_d += np.logaddexp(topic_model,bg_model)
        # P_q_d += np.logaddexp(np.logaddexp(unigram_model,topic_model),bg_model)
    return P_q_d

def getSimilarity(a, b, queryIDs, docTF, docLength, docUnigram, bg, P_T_d, P_w_T):
    querysSim=[]
    for q in tqdm(range(len(queryIDs))):
        query = queryIDs[q]
        bg_i = [bg[i] for i in range(len(query))]
        P_wi_T = [P_w_T[:,query[i]] for i in range(len(query))]
        
        sim = []
        for j in range(len(docLength)):
            P_wi_dj = [docUnigram[j,i] for i in range(len(query))]
            P_T_dj = P_T_d[j,:]
            sim.append(calculateP_q_d(a, b, len(query), P_wi_dj, bg_i, P_T_dj, P_wi_T))
        
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True)[:1000])
    return querysSim

