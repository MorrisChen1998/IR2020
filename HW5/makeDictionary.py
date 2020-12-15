# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:30:17 2019

@author: morri
"""

def createDictionary(querys,docs):
    querysWords=[]
    docsWords=[]
    dictionary=[]
    for query in querys:
        words=query.split(' ')
        querysWords.append(words)
        for word in words:
            if word not in dictionary and word != '':
                dictionary.append(word)

    for doc in docs:
        words=doc.split(' ')
        docsWords.append(words)
#        for word in words:
#            if word not in dictionary and word != '':
#                dictionary.append(word)
                
    return dictionary,querysWords,docsWords
