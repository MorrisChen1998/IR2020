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
# start_time = time.time()
# import calculateTF
# print('......calculating TF......')
# queryTF = calculateTF.getQueryTF(dictionary,querysWords)
# docTF = calculateTF.getDocumentTF(dictionary,docsWords)
# print("%.2f" % (time.time() - start_time))

# #%% calculate IDF
# start_time = time.time()
# import calculateIDF
# print('......calculating IDF......')
# idf = calculateIDF.getIDF(queryTF, docTF, dictionary)
# print("%.2f" % (time.time() - start_time))
#%%
import pickle
with open("querysSim.txt", "rb") as fp:   # Unpickling
    querysSim = pickle.load(fp)
print("%.2f" % (time.time() - start_time))
#%%
import queryPrefix
fix=5
dictionary_fix = queryPrefix.createDictionary(querysWords,docsWords,querysSim,fix)
print('......calculating TF......')
start_time = time.time()
import calculateTF
queryTF = calculateTF.getQueryTF(dictionary_fix,querysWords)
docTF = calculateTF.getDocumentTF(dictionary_fix,docsWords)
print("%.2f" % (time.time() - start_time))

#%
import numpy as np
np.save('queryTF',queryTF)
np.save('docTF',docTF)
with open("dictionary_fix5.txt", "wb") as fp:   #Pickling
    pickle.dump(dictionary_fix, fp)
    
#%%
with open("dictionary_fix5.txt", "rb") as fp:   # Unpickling
    dictionary_fix = pickle.load(fp)

import numpy as np
print('......loading queryTF......')
queryTF = np.load('queryTF.npy')
print('......loading docTF......')
docTF = np.load('docTF.npy')

print("%.2f" % (time.time() - start_time))
print('......calculating IDF......')
import calculateIDF
start_time = time.time()
idf = calculateIDF.getIDF(queryTF, docTF, dictionary_fix)
print("%.2f" % (time.time() - start_time))

#%%
#rocchio
fix=5
n_fix=5
alpha=1
beta=0.5
gamma=0.3
epochs=1
#bm25
a1 = 0.6
a2 = 0.3
k1 = 1
k3 = 1000
b = 1
delta = .75
import queryPrefix
queryTF_new = queryTF
for i in range(epochs):
    queryTF_new = queryPrefix.queryFix(queryTF_new,docTF,querysSim,alpha,beta,gamma,fix,n_fix)

#% tuning model parameters and calculate BM25
start_time = time.time()
import calculateBM26
print('......calculating BM25......')

querysSim_new = calculateBM26.getSimilarity(a1, a2, k1, k3, b, delta, documentLengthNormalizations,queryTF_new, docTF, idf)
print("%.2f" % (time.time() - start_time))

#% print out
start_time = time.time()
import printOutAnswer
print('......print out answer......')
printOutAnswer.printOutAnswer(queryList, docList, querysSim_new)
print("%.2f" % (time.time() - start_time))