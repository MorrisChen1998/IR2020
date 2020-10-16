# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:16:46 2019

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

#%% calculate TF-IDF
start_time = time.time()
import calculateTFIDF
print('......calculating TF-IDF......')
queryTFIDF = calculateTFIDF.getQueryTFIDF(queryTF,IDF)
docTFIDF = calculateTFIDF.getDocTFIDF(docTF,IDF)
print(time.time() - start_time)

#%% calculate similarity
start_time = time.time()
import calculateCos
print('......calculating similarity between queries and documents......')
querysSim = calculateCos.getSimilarity(docTFIDF, queryTFIDF)
print(time.time() - start_time)

#%% print out
start_time = time.time()
import printOutAnswer
print('......print out answer......')
printOutAnswer.printOutAnswer(queryList, docList, querysSim)
print(time.time() - start_time)
