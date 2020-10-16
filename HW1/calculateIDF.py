# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:40:17 2019

@author: morri
"""

import math
import numpy as np

def getIDF(queryTF, docTF, dictionary):
    docTFinverse=docTF.transpose()
    queryTFinverse=queryTF.transpose()
    IDF = []
    for index in range(len(dictionary)):
        IDF.append(math.log(1+(len(docTFinverse[index])+len(queryTFinverse[index]))/(np.count_nonzero(queryTFinverse[index])+np.count_nonzero(docTFinverse[index]))))
    return np.array(IDF)