# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:37:47 2019

@author: morri
"""

import numpy as np
from tqdm import tqdm

def getQueryTF(dictionary, querysWords):
    queryTF = []
    for q in tqdm(range(len(querysWords))):
        queryTFj = []
        for word in dictionary:
           queryTFj.append(querysWords[q].count(word))
        queryTF.append(queryTFj)
    return np.array(queryTF)

def getDocumentTF(dictionary, docsWords):
    docTF = []
    for j in tqdm(range(len(docsWords))):
        docTFj = []
        for word in dictionary:
            docTFj.append(docsWords[j].count(word))
        docTF.append(docTFj)    
    return np.array(docTF)
