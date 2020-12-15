# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:55:05 2019

@author: morri
"""
import numpy as np
def getDocumentLengthNormalization(docsWords):
    average = 0
    documentLengthNormalization = []
    for doc in docsWords:
        average+=len(doc)/len(docsWords)
        documentLengthNormalization.append(len(doc))

    documentLengthNormalization = np.array(documentLengthNormalization)/average
    return documentLengthNormalization