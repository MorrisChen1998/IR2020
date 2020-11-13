# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:37:47 2019

@author: morri
"""

import numpy as np

def getQueryTF(dictionary, querysWords):
    queryTF = []
    for queryWords in querysWords:
        queryTFj = []
        for word in dictionary:
           queryTFj.append(queryWords.count(word))
        queryTF.append(queryTFj)
    return np.array(queryTF)

def getDocumentTF(dictionary, docsWords):
    docTF = []
    for docWords in docsWords:
        docTFj = []
        for word in dictionary:
            docTFj.append(docWords.count(word))
        docTF.append(docTFj)    
    return np.array(docTF)
