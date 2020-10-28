# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:40:17 2019

@author: morri
"""

import numpy as np

def getIDF(queryTF, docTF, dictionary):
    docTFinverse=docTF.transpose()
    queryTFinverse=queryTF.transpose()
    IDF = []
    for index in range(len(dictionary)):
        N = len(docTFinverse[index])#+len(queryTFinverse[index])
        ni = np.count_nonzero(docTFinverse[index])#+np.count_nonzero(queryTFinverse[index])
        IDF.append(np.log10((N-ni+0.5)/(ni+0.5)))
    return np.array(IDF)