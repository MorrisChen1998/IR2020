# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:37:41 2019

@author: morri
"""
def importData():
    querys=[]
    docs=[]
        
    queryListDoc = open("query_list.txt",'r')
    docListDoc = open("doc_list.txt",'r')
    queryList=[]
    docList=[]
        
    for query in queryListDoc:
        queryList.append(query.strip())
        
    for doc in docListDoc:
        docList.append(doc.strip())
            
    queryListDoc.close()
    docListDoc.close()
        
    for query in queryList:
        queryDoc=open("queries/"+query+".txt",'r')
        querys.append(queryDoc.read())
        queryDoc.close()
    for doc in docList:
        docDoc=open("docs/"+doc+".txt",'r')
        docs.append(docDoc.read())
        docDoc.close()
        
    return queryList, docList, querys, docs
