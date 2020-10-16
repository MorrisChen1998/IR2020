# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:30:17 2019

@author: morri
"""
import tqdm

querysWords={}
docsWords={}
dictionary=[]

def createDictionary(querys,docs):
    for query in tqdm(querys):
        words=querys[query].split(' ')
        querysWords[query]=words
        for word in words:
            if word not in dictionary and word != '':
                dictionary.append(word)

    for doc in tqdm(docs):
        words=docs[doc].split(' ')
        docsWords[doc]=words
        for word in words:
            if word not in dictionary and word != '':
                dictionary.append(word)
                
    return dictionary;

def getQuerysWords():
    return querysWords;
def getDocsWords():
    return docsWords;
