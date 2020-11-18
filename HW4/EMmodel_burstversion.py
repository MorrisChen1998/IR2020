import numpy as np
print('......EM training......')
wNumber = 111499#len(dictionary)
dNumber = 14955#len(docs)
topicNum = 30
# P_w_T = np.random.dirichlet(np.ones(n_word),size= n_topic)
# P_T_d = np.random.dirichlet(np.ones(n_topic),size= n_doc)
P_w_T = np.ones((topicNum, wNumber))/wNumber
P_T_d = np.ones((dNumber, topicNum))/topicNum
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
            if docLength[j] == 0:
                P_T_d[j,k] = 1.0 / dNumber
            else:
                P_T_d[j,k] /= docLength[j]

def LogLikelihood():
    loglikelihood = 0
    for i in range(wNumber):
        for j in range(dNumber):
            loglikelihood += docTF[j,i]*np.log10(np.dot(P_T_d[j,:],P_w_T[:,i])/dNumber)
    return loglikelihood

for train in range(1):
    start_time = time.time()
    Estep()
    Mstep()
    print("epoch %d, cost time %.2fs, loglikelihood = %.2f" % (train + 1,time.time() - start_time, LogLikelihood()))

