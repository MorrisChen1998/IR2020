# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:49:25 2020

@author: morri
"""
import numpy as np
from numba import jit
from tqdm import tqdm

#%%
@jit(nopython=True)#with numba to accelerate
def em_step(P_smm, bg, sum_Reldoc, alpha):
    #e step
    Tsmm = (1-alpha)*P_smm/((1-alpha)*P_smm+alpha*bg)
    #m step
    P_smm = sum_Reldoc*Tsmm
    P_smm = P_smm/np.sum(P_smm)
    return P_smm
 
#%%
@jit(nopython=True)#with numba to accelerate
def KLdivergence(docUnigram_row, docUnigram_col, docUnigram_val, P_smm, bg, query, querySim, a, b, r):
    P_w_q = np.zeros(len(bg))
    for w in range(len(bg)):
        P_w_q[w] = b*P_smm[w]+(1-a-b)*bg[w]
    for w in range(len(query)):
        P_w_q[query[w]]+=a/len(query)
        
    nnz = len(docUnigram_val)
    sim = np.zeros(len(querySim))
    for i in range(nnz):
        row, col = docUnigram_row[i], docUnigram_col[i]
        sim[row] += np.log(1+docUnigram_val[i]*r+bg[col]*(1-r))*P_w_q[col]
            
    return sim

#%%
def SimpleMixtureModel(docTF, docUnigram, docLength, bg, queryIDs, querysSim, n_iter, alpha, topK, a, b, r):
    new_querysSim=[]
    #for q in tqdm(range(1)):
    for q in tqdm(range(len(querysSim))):
        sum_Reldoc=np.zeros(len(bg))
        for i in range(topK):
            sum_Reldoc = sum_Reldoc + docTF[querysSim[q][i]].toarray()
        P_smm = np.random.dirichlet(np.ones(len(bg)))
        for epoch in range(n_iter):
            P_smm = em_step(P_smm,bg,sum_Reldoc,alpha)
        P_smm = np.squeeze(P_smm)
        sim = KLdivergence(docUnigram.row,docUnigram.col,docUnigram.data,P_smm,bg,np.array(queryIDs[q]),np.array(querysSim[q]),a,b,r)
        new_querysSim.append(sorted(range(len(sim)), key=lambda k: sim[k], reverse = True))
    return new_querysSim
