# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:40:32 2020

@author: morri
"""
import time
import numpy as np

#%% load file
start_time = time.time()
import importData
print('......loading file......')
queryList, docList, querys, docs = importData.importData()
print("%.2f" % (time.time() - start_time))

#%% make dictionary
start_time = time.time()
print('......document preprocessing......')
import documentPreprocessing
dictionary,docLength,docTF, docUnigram = documentPreprocessing.createDictionary(docs)
queryIDs = documentPreprocessing.getQueryID(dictionary, querys)
print("%.2f" % (time.time() - start_time))
#%% calculate BG
start_time = time.time()
# print('......calculating BG......')
# import calculateBG
# bg = calculateBG.getBG(dictionary,docTF,docLength)
# np.save('bg',bg)
'''
BG model cost half hour to train
'''
print('......loading BG......')
bg = np.load('bg.npy')
print("%.2f" % (time.time() - start_time))

#%%
n_topic = 50
n_iter = 100
start_time = time.time()
print('......training PLSA model......')
import calculatePLSA as plsa
P_T_d, P_w_T = plsa.plsa_training(docTF.row, docTF.col, docTF.data, len(dictionary), len(docLength), n_topic, n_iter)
loglikelihood = plsa.calculateLikelihood(docTF, P_T_d, P_w_T, len(docLength))
#%% calculate similarity
start_time = time.time()
print('......calculating similarity......')
a = 0.1
b = 0.8
querysSim = plsa.getSimilarity(a, b, queryIDs, docTF, docLength, docUnigram, bg, P_T_d, P_w_T)
print("%.2f" % (time.time() - start_time))

#%% print out
start_time = time.time()
print('......print out answer......')
import printOutAnswer
printOutAnswer.printOutAnswer(queryList, docList, querysSim)
print("%.2f" % (time.time() - start_time))