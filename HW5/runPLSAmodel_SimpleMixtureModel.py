# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:40:32 2020

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

#% preprocessing
start_time = time.time()
print('......document preprocessing......')
import documentPreprocessing
dictionary,docLength,docTF, docUnigram = documentPreprocessing.createDictionary(docs)
queryIDs = documentPreprocessing.getQueryID(dictionary, querys)
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
print("%.2f" % (time.time() - start_time))

#%%
n_topic = 32
n_iter = 100
start_time = time.time()
print('......training PLSA model......')
import calculateEMtraining as em
docTF=docTF.tocoo()
P_T_d, P_w_T = em.plsa_training(docTF.row, docTF.col, docTF.data, len(dictionary), len(docLength), n_topic, n_iter)
# loglikelihood = em.calculateLikelihood(docTF, P_T_d, P_w_T, len(docLength))

#% calculate similarity
start_time = time.time()
print('......calculating similarity......')
import calculatePLSA as plsa
a = 0.7
b = 0.3
querysSim = plsa.getSimilarity(a, b, queryIDs, docUnigram, bg, P_T_d, P_w_T)
print("%.2f" % (time.time() - start_time))

#%%
# with open("querysSim.txt", "wb") as fp:   #Pickling
#     pickle.dump(querysSim, fp)
with open("querysSim.txt", "rb") as fp:   # Unpickling
    querysSim = pickle.load(fp)
    
#%%
start_time = time.time()
import calculateSMM as smm
n_iter=10
alpha=0.5
topK=3
a=0.64  
b=0.35
r=0.99
querysSim_new=smm.SimpleMixtureModel(docTF, docUnigram.tocoo(), docLength, bg, queryIDs, querysSim, n_iter, alpha, topK, a, b, r)
print("%.2f" % (time.time() - start_time))

#% print out
start_time = time.time()
print('......print out answer......')
import printOutAnswer
printOutAnswer.printOutAnswer(queryList, docList, querysSim_new)
print("%.2f" % (time.time() - start_time))