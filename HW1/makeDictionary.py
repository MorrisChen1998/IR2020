# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:30:17 2019

@author: morri
"""
import re

def createDictionary(querys,docs):
    querysWords=[]
    docsWords=[]
    dictionary=[]
    for query in querys:
        query = re.sub(r'[0-9]', '', query)
        words=query.split(' ')
        querysWords.append(words)
        for word in words:
            if word not in dictionary and word != '':
                dictionary.append(word)

    for doc in docs:
        doc = re.sub(r'[0-9]', '', doc)
        words=doc.split(' ')
        docsWords.append(words)
        for word in words:
            if word not in dictionary and word != '':
                dictionary.append(word)
                
    return dictionary,querysWords,docsWords
