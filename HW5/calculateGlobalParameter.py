# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:55:05 2020

@author: morri
"""
from tqdm import tqdm
import numpy as np

#%%
def getBG(dictionary,docTF,docLength):
    BG=[]
    docTFinverse=docTF.toarray().transpose()
    corpus_length = np.sum(docLength)
    for i in tqdm(range(len(dictionary))):
        BG.append(np.sum(docTFinverse[i]) / corpus_length)
    
    return np.array(BG)

#%%
def getIDF(docTF, n_word, N):
    docTFinverse=docTF.toarray().transpose()
    IDF = []
    for i in tqdm(range(n_word)):
        ni = np.count_nonzero(docTFinverse[i])
        IDF.append(np.log(1+(N-ni+0.5)/(ni+0.5)))
    return np.array(IDF)
