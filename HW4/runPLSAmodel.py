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
queryIDs = calculateTF.getQueryID(dictionary,querysWords)
docTF = calculateTF.getDocumentTF(dictionary,docsWords)
print("%.2f" % (time.time() - start_time))

#%% calculate BG
start_time = time.time()
import calculateBGandUnigram
print('......calculating BG and Unigram......')
documentLength, BG, P_w_d = calculateBGandUnigram.getBGandUnigram(dictionary,docsWords,docTF)
print("%.2f" % (time.time() - start_time))

#%% EM model initialize
import sparse, numpy as np
print('......EM training......')
wNumber = len(dictionary)
dNumber = len(docs)
topicNum = 30
P_w_T = np.random.dirichlet(np.ones(wNumber),size= topicNum)
P_T_d = np.random.dirichlet(np.ones(topicNum),size= dNumber)
#P_w_T = np.ones((topicNum, wNumber))/wNumber
#P_T_d = np.ones((dNumber, topicNum))/topicNum
P_T_wd = np.zeros(shape=(wNumber, dNumber, topicNum))
P_T_wd = sparse.COO(P_T_wd)
def Estep():
    global P_T_wd
    for i in range(wNumber):
        for j in range(dNumber):
            P_T_wd[i,j,:] = P_w_T[:,i]*P_T_d[j,:]
            sumP_w_T_P_T_d = np.sum(P_T_wd[i,j,:])
            
            if sumP_w_T_P_T_d == 0:
                P_T_wd[i,j,:] = np.zeros(shape=(topicNum))
            else:
                P_T_wd[i,j,:] /= sumP_w_T_P_T_d;
           
def Mstep():
    global P_w_T
    global P_T_d
    for k in range(topicNum):
        sumC_w_d_P_T_wd = 0
        for i in range(wNumber):
            P_w_T[k,i] = np.dot(docTF[:,i],P_T_wd[i,:,k])
            sumC_w_d_P_T_wd += P_w_T[k,i]
        
        if sumC_w_d_P_T_wd == 0:
            P_w_T[k,:] = np.ones(wNumber)/wNumber
        else:
            P_w_T[k,:] /= sumC_w_d_P_T_wd
        
        for j in range(dNumber):
            P_T_d[j,k] = np.dot(P_T_wd[:,j,k],docTF[j,:])
            if documentLength[j] == 0:
                P_T_d[j,k] = 1.0 / dNumber
            else:
                P_T_d[j,k] /= documentLength[j]

def LogLikelihood():
    loglikelihood = 0
    for i in range(wNumber):
        for j in range(dNumber):
            loglikelihood += docTF[j,i]*np.log10(np.dot(P_T_d[j,:],P_w_T[:,i])/dNumber)
    return loglikelihood
#%% EM train start
for train in range(30):
    start_time = time.time()
    Estep()
    Mstep()
    print("epoch %d, cost time %.2fs, loglikelihood = %.2f" % (train + 1,time.time() - start_time, LogLikelihood()))

#%% calculate similarity
start_time = time.time()
import calculatePLSAsimilarity as plsa
print('......calculating PLSA similarity......')
a = 0.3
b = 0.5
querysSim = plsa.getSimilarity(a, b, dictionary, queryIDs, docTF, P_w_d, BG, P_T_d, P_w_T)
print("%.2f" % (time.time() - start_time))

#%% print out
start_time = time.time()
import printOutAnswer
print('......print out answer......')
printOutAnswer.printOutAnswer(queryList, docList, querysSim)
print("%.2f" % (time.time() - start_time))