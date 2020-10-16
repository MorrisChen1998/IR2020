# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:39:45 2019

@author: morri
"""

def printOutAnswer(queryList, docList, querysSim):
    file= open("answer.txt","w+")
    file.write("Query,RetrievedDocuments\n")
    for query in range(len(querysSim)):
        file.write("%s,"%queryList[query])
        for doc in range(len(querysSim[query])):
            file.write(" %s"%docList[querysSim[query][doc]])
        file.write("\n")
    file.close()