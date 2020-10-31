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
print("%.2f" % (time.time() - start_time))

#%% make dictionary
start_time = time.time()
import makeDictionary
print('......making the dictionary......')
dictionary,querysWords,docsWords = makeDictionary.createDictionary(querys,docs)
print("%.2f" % (time.time() - start_time))

#%% calculate document length normalization
start_time = time.time()
import calculateDocLenNorm
print('......calculating dlen norm......')
documentLengthNormalizations = calculateDocLenNorm.getDocumentLengthNormalization(docsWords)
print("%.2f" % (time.time() - start_time))

#%% calculate TF
start_time = time.time()
import calculateTF
print('......calculating TF......')
queryTF = calculateTF.getQueryTF(dictionary,querysWords)
docTF = calculateTF.getDocumentTF(dictionary,docsWords)
print("%.2f" % (time.time() - start_time))

#%% calculate IDF
start_time = time.time()
import calculateIDF
print('......calculating IDF......')
idf = calculateIDF.getIDF(queryTF, docTF, dictionary)
print("%.2f" % (time.time() - start_time))

#%% tuning model parameters and calculate BM25
start_time = time.time()
import calculateBM25
print('......calculating BM25......')
# k3 is useless parameter for short query
k3 = 3

a1 = .7
a2 = .1
k1 = 2.5
b = .8
delta = .75

querysSim = calculateBM25.getSimilarity(a1, a2, k1, k3, b, delta, documentLengthNormalizations,queryTF, docTF, idf)
print("%.2f" % (time.time() - start_time))

#% print out
start_time = time.time()
import printOutAnswer
print('......print out answer......')
printOutAnswer.printOutAnswer(queryList, docList, querysSim)
print("%.2f" % (time.time() - start_time))