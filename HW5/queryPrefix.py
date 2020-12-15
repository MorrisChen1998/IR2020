# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

def createDictionary(querysWords,docsWords,querysSim,fix):
    dictionary_fix=[]
    for q in tqdm(range(len(querysWords))):
        for word in querysWords[q]:
            if word not in dictionary_fix and word != '':
                dictionary_fix.append(word)

        for j in range(fix):
            for word in docsWords[querysSim[q][j]]:
                if word not in dictionary_fix and word != '':
                    dictionary_fix.append(word)
                
    return dictionary_fix

def queryFix(queryTF,docTF,querysSim,alpha,beta,gamma,fix,n_fix):
    new_queryTF=[]
    for q in range(len(queryTF)):
        relevant_doc = np.sum([docTF[querysSim[q][i]]/fix for i in range(fix)],axis=0)
        nonrelevant_doc = np.sum([docTF[querysSim[q][n_fix-i-1]]/n_fix for i in range(n_fix)],axis=0)        
        new_queryTF.append(queryTF[q]*alpha+relevant_doc*beta-gamma*nonrelevant_doc)

    return np.array(new_queryTF)