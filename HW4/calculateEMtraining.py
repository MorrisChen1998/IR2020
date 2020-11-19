# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from tqdm import tqdm

#%%
def calculateLikelihood(docTF, P_T_d, P_w_T, doc_len):
    loglikelihood = 0
    for i in tqdm(range(len(docTF.data))):
        loglikelihood += docTF.data[i]*np.log10(np.dot(P_T_d[docTF.row[i],:],P_w_T[:,docTF.col[i]]))
    return loglikelihood

#%%
def plsa_training(docTF_row, docTF_col, docTF_val, idf, n_word, n_doc, n_topic, n_iter):
    P_w_T = np.random.dirichlet(np.ones(n_word),size= n_topic)
    P_T_d = np.random.dirichlet(np.ones(n_topic),size= n_doc)
    for epoch in tqdm(range(n_iter)):
        P_T_d, P_w_T = em_step(docTF_row, docTF_col, docTF_val, idf, P_T_d, P_w_T, n_word, n_doc, n_topic)
    return P_T_d, P_w_T

#%%
@jit(nopython=True)#with numba to accelerate
def em_step(docTF_row, docTF_col, docTF_val, idf, P_T_d, P_w_T, n_word, n_doc, n_topic):
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
            q = docTF_val[i] * idf[docTF_col[i]] * sparseP_T_w_d[i, k]
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