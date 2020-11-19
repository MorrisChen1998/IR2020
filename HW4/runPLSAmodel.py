# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:40:32 2020

@author: morri
"""
import time
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
dictionary,docLength,docTF, docUnigram = documentPreprocessing.createDictionary(docs)
queryIDs = documentPreprocessing.getQueryID(dictionary, querys)
print("%.2f" % (time.time() - start_time))

#%% calculate global parameter
start_time = time.time()
# print('......calculating BG......')
# from calculateGlobalParameter import getBG
# bg = calculateBG.getBG(dictionary,docTF,docLength)
# np.save('bg',bg)
'''
BG model cost half hour to train
'''
# print('......calculating idf......')
# from calculateGlobalParameter import getIDF
# idf = getIDF(docTF, len(dictionary), len(docLength))
# np.save('idf',idf)
print('......loading BG and idf......')
bg = np.load('bg.npy')
idf = np.load('idf.npy')
print("%.2f" % (time.time() - start_time))

#%%
n_topic = 128
n_iter = 100
start_time = time.time()
print('......training PLSA model......')
import calculateEMtraining as em
P_T_d, P_w_T = em.plsa_training(docTF.row, docTF.col, docTF.data, idf, len(dictionary), len(docLength), n_topic, n_iter)
# loglikelihood = em.calculateLikelihood(docTF, P_T_d, P_w_T, len(docLength))

#%% calculate similarity
start_time = time.time()
print('......calculating similarity......')
import calculatePLSA as plsa
a = 0.6
b = 0.4
querysSim = plsa.getSimilarity(a, b, queryIDs, docUnigram, bg, P_T_d, P_w_T)
print("%.2f" % (time.time() - start_time))

#% print out
start_time = time.time()
print('......print out answer......')
import printOutAnswer
printOutAnswer.printOutAnswer(queryList, docList, querysSim)
print("%.2f" % (time.time() - start_time))