# -*- coding: utf-8 -*-
from tqdm import tqdm
from scipy.sparse import csr_matrix

#%%
def createDictionary(docs):
    docLength=[]
    indptr = [0]
    indices = []
    tf_data = []
    unigram_data = []
    dictionary = {}
    
    for j in tqdm(range(len(docs))):
        words=docs[j].split()
        docLength.append(len(words))
        for word in words:
            index = dictionary.setdefault(word, len(dictionary))
            indices.append(index)
            tf_data.append(1)
            unigram_data.append(1/len(words))
        indptr.append(len(indices))
        
    docTF=csr_matrix((tf_data, indices, indptr), dtype=int).tocoo()
    docUnigram=csr_matrix((unigram_data, indices, indptr), dtype=float)
    '''
    docTF.getrow(x)=x document
    docTF.getcol(x)=x term freq
    '''
    return dictionary, docLength, docTF, docUnigram

#%%
def getQueryID(dictionary, querys):
    queryID = []
    for query in querys:
        words=query.split()
        ids = []
        for word in words:
            ids.append(dictionary[word])
        queryID.append(ids)
    return queryID