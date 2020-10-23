# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:37:47 2019

@author: morri
"""

import numpy as np
import math

def logNormalization(tf):
    return math.log(tf+1,2)

def getQueryTF(dictionary, querysWords):
    queryTF = []
    for queryWords in querysWords:
        queryTFj = []
        for word in dictionary:
            queryTFj.append(logNormalization(queryWords.count(word)))
        queryTF.append(queryTFj)    
    return np.array(queryTF)

def getDocumentTF(dictionary, docsWords):
    docTF = []
    for docWords in docsWords:
        docTFj = []
        for word in dictionary:
            docTFj.append(logNormalization(docWords.count(word)))
        docTF.append(docTFj)    
    return np.array(docTF)
