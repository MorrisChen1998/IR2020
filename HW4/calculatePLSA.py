# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from tqdm import tqdm

def calculateLikelihood(docTF, P_T_d, P_w_T, doc_len):
    loglikelihood = 0
    for i in tqdm(range(len(docTF.data))):
        loglikelihood += docTF.data[i]*np.log10(np.dot(P_T_d[docTF.row[i],:],P_w_T[:,docTF.col[i]])/doc_len)
    return loglikelihood

def plsa_training(docTF_row, docTF_col, docTF_val, n_word, n_doc, n_topic, n_iter):
    P_w_T = np.random.dirichlet(np.ones(n_word),size= n_topic)
    P_T_d = np.random.dirichlet(np.ones(n_topic),size= n_doc)
    for i in tqdm(range(n_iter)):
        P_T_d, P_w_T = em_step(docTF_row, docTF_col, docTF_val, P_T_d, P_w_T, n_word, n_doc, n_topic)
    return P_T_d, P_w_T

@jit(nopython=True)#with numba to accelerate
def em_step(docTF_row, docTF_col, docTF_val, P_T_d, P_w_T, n_word, n_doc, n_topic):
    nnz = len(docTF_val)

    sparseP_T_w_d = np.zeros((nnz, n_topic))
    # E step
    for i in range(nnz):
        P_w_T_P_T_d = np.zeros((n_topic))
        d, t = docTF_row[i], docTF_col[i]
        sigma = 0
        for k in range(n_topic):
            P_w_T_P_T_d[k] = P_T_d[d, k] * P_w_T[k, t]
            sigma += P_w_T_P_T_d[k]
        for k in range(n_topic):
            sparseP_T_w_d[i, k] = P_w_T_P_T_d[k] / sigma
    
    # M step
    P_T_d[:] = 0
    P_w_T[:] = 0
    w_sum = np.zeros((n_topic))
    d_sum = np.zeros((n_doc))
    for i in range(nnz):
        for k in range(n_topic):
            q = docTF_val[i] * sparseP_T_w_d[i, k]
            P_w_T[k, docTF_col[i]] += q
            w_sum[k] += q
            P_T_d[docTF_row[i], k] += q
            d_sum[docTF_row[i]] += q
        
    # Normalize P(T|d)
    for j in range(n_doc):
        for k in range(n_topic):
            P_T_d[j, k] /= d_sum[j]
    # Normalize P(w|T)
    for k in range(n_topic):
        for i in range(n_word):
            P_w_T[k, i] /= w_sum[k]
            
    return P_T_d, P_w_T

def calculateP_q_d(a, b, query_len, P_wi_dj, bg_i, P_T_dj, P_wi_T):
    P_q_d = 0
    for i in range(query_len):
        unigram_model = np.log(a)+np.log(P_wi_dj[i] if P_wi_dj[i]>1e-50 else 1e-50)
        topic_model = np.log(b)+np.log(np.dot(P_T_dj,P_wi_T[i]))
        bg_model = np.log(1-a-b)+np.log(bg_i[i] if bg_i[i]>1e-50 else 1e-50)
        P_q_d += np.logaddexp(np.logaddexp(unigram_model,topic_model),bg_model)
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

