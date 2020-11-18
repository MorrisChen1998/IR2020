# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

#%%
def calculatelogP_q_d(a, b, query_len, P_wi_dj, bg_i, P_T_dj, P_wi_T):
    P_q_d = 0
    for i in range(query_len):
        unigram_model = np.log(a)+np.log(P_wi_dj[i])
        topic_model = np.log(b)+np.log(np.dot(P_T_dj,P_wi_T[i]))
        bg_model = np.log(1-a-b)+np.log(bg_i[i])
        # P_q_d += np.logaddexp(topic_model,bg_model)
        P_q_d += np.logaddexp(np.logaddexp(unigram_model,topic_model),bg_model)
    return P_q_d

#%%
def calculateP_q_d(a, b, query_len, P_wi_dj, bg_i, P_T_dj, P_wi_T):
    P_q_d = 0
    for i in range(query_len):
        unigram_model = a*P_wi_dj[i]
        topic_model = b*np.dot(P_T_dj, P_wi_T[i])
        bg_model = (1-a-b)*bg_i[i]
        P_q_d += np.log(unigram_model+topic_model+bg_model)
    return P_q_d

#%%
def getSimilarity(a, b, queryIDs, docUnigram, bg, P_T_d, P_w_T):
    querysSim=[]
    for q in tqdm(range(len(queryIDs))):
        query = queryIDs[q]
        bg_i = [bg[i] for i in range(len(query))]
        P_wi_T = [P_w_T[:,query[i]] for i in range(len(query))]
        
        sim = []
        for j in range(len(P_T_d)):
            P_wi_dj = [docUnigram[j,i] for i in range(len(query))]
            P_T_dj = P_T_d[j,:]
            sim.append(calculateP_q_d(a, b, len(query), P_wi_dj, bg_i, P_T_dj, P_wi_T))
        
        querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True)[:1000])
    return querysSim

