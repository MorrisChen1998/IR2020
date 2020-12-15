# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:16:08 2020

@author: morri
"""
import time
import pickle
import numpy as np

#% load file
start_time = time.time()
import importData
print('......loading file......')
queryList, docList, querys, docs  = importData.importData()
print("%.2f" % (time.time() - start_time))

#%% preprocessing
start_time = time.time()
print('......document preprocessing......')
import documentPreprocessing
dictionary,docLength,docTF = documentPreprocessing.createDictionary(docs)
queryIDs = documentPreprocessing.getQueryID(dictionary, querys)
DLNs = documentPreprocessing.getDLN(docLength)
print("%.2f" % (time.time() - start_time))

#% calculate global parameter
start_time = time.time()
# print('......calculating BG......')
# from calculateGlobalParameter import getBG
# bg = getBG(dictionary,docTF,docLength)
# np.save('bg',bg)
# print('......calculating idf......')
# from calculateGlobalParameter import getIDF
# idf = getIDF(docTF, len(dictionary), len(docLength))
# np.save('idf',idf)
print('......loading BG......')
bg = np.load('bg.npy')
print('......loading idf......')
idf = np.load('idf.npy')
#%%
# with open("querysSim.txt", "wb") as fp:   #Pickling
#     pickle.dump(querysSim, fp)
with open("querysSim.txt", "rb") as fp:   # Unpickling
    querysSim = pickle.load(fp)
print("%.2f" % (time.time() - start_time))

#%%
start_time = time.time()
print('......calculating second IR......')
# import calculateCos
import calculateBM25
a1 = .7
a2 = .1
k1 = 2
k3 = 0.5
b = 0.7
delta = .75

topK=1
alpha=3
beta=1
# querysSim=calculateCos.getSimilarity(docTF,queryIDs,querysSim,idf,topK,alpha,beta)
querysSim_new=calculateBM25.getSimilarity(a1, a2, k1, k3, b, delta, DLNs, docTF,queryIDs,querysSim,idf,topK,alpha,beta)
print("%.2f" % (time.time() - start_time))

#% print out
start_time = time.time()
print('......print out answer......')
import printOutAnswer
printOutAnswer.printOutAnswer(queryList, docList, querysSim_new)
print("%.2f" % (time.time() - start_time))