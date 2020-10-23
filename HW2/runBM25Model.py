# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 21:45:32 2019

@author: morri
"""

#%% load file
import time
start_time = time.time()
import importData
print('......loading file......')
queryList, docList, querys, docs = importData.importData()
print(time.time() - start_time)

#%% make dictionary
start_time = time.time()
import makeDictionary
print('......making the dictionary......')
dictionary,querysWords,docsWords = makeDictionary.createDictionary(querys,docs)
print(time.time() - start_time)

#%% calculate document length normalization
start_time = time.time()
import calculateDocLenNorm
print('......calculating document length normalization......')
documentLengthNormalizations = calculateDocLenNorm.getDocumentLengthNormalization(docsWords)
print(time.time() - start_time)

#%% calculate TF
start_time = time.time()
import calculateTF
print('......calculating TF......')
queryTF = calculateTF.getQueryTF(dictionary,querysWords)
docTF = calculateTF.getDocumentTF(dictionary,docsWords)
print(time.time() - start_time)

#%% calculate IDF
start_time = time.time()
import calculateIDF
print('......calculating IDF......')
IDF = calculateIDF.getIDF(queryTF, docTF, dictionary)
print(time.time() - start_time)

#%% tuning model parameters and calculate BM25
start_time = time.time()
import calculateBM25
print('......calculating BM25......')
k1 = 1.5
k3 = 5
b = 1
delta = 0.8
querysSim = calculateBM25.getSimilarity(k1, k3, b, delta, documentLengthNormalizations,queryTF, docTF, IDF)
print(time.time() - start_time)

#%% print out
start_time = time.time()
import printOutAnswer
print('......print out answer......')
printOutAnswer.printOutAnswer(querysSim)
print(time.time() - start_time)
