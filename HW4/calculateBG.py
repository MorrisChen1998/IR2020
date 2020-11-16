# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:55:05 2020

@author: morri
"""
from tqdm import tqdm
import numpy as np
def getBG(dictionary,docTF,docLength):
    BG=[]
    corpus_length = np.sum(docLength)
    for i in tqdm(range(len(dictionary))):
        BG.append(np.sum(docTF.getcol(i)) / corpus_length)
    
    return np.array(BG)