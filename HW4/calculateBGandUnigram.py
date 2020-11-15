# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:55:05 2019

@author: morri
"""
import numpy as np
def getBGandUnigram(dictionary,docsWords,docTF):
    #doc length
    documentLength = []
    for j in range(len(docTF)):
        documentLength.append(np.sum(docTF[j]))
    documentLength = np.array(documentLength)
    corpus_length = np.sum(documentLength)
    
    #BG
    BG=[]
    for i in range(len(dictionary)):
        BG.append(np.sum(docTF.transpose()[i]) / corpus_length)
    np.array(BG)
    
    #Unigram c_wi_dj/len(dj)
    P_w_d = np.transpose(docTF.transpose() / documentLength)
    
    return documentLength, BG, P_w_d