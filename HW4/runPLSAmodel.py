# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:40:32 2020

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

#%% calculate TF
start_time = time.time()
import calculateTF
print('......calculating TF......')
queryTF = calculateTF.getQueryTF(dictionary,querysWords)
docTF = calculateTF.getDocumentTF(dictionary,docsWords)
print("%.2f" % (time.time() - start_time))