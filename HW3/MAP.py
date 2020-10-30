data = ['s1','s2','s3']
for x in range(3):
    f = open(data[x]+'.txt')
    query = int(f.readline())
    MAP = float(0)
    for i in range(query):
        q_r = f.readline().strip('\n').split(' ')
        q_a = f.readline().strip('\n').split(' ')
        hit = float(0)
        for d in range(len(q_r)):
            if(q_r[d] in q_a):
                hit+=1
                MAP+=hit/((d+1)*len(q_a)*query)
            if(hit>=len(q_a)):
                break
    f.close()
    print("%.4f" % MAP)

#%%
query = int(input())
MAP = float(0)
for i in range(query):
    q_r = input().split(' ')
    q_a = input().split(' ')
    hit = float(0)
    
    for d in range(len(q_r)):
        if(q_r[d] in q_a):
            hit+=1
            MAP+=hit/((d+1)*len(q_a)*query)
        if(hit>=len(q_a)):
            break
print("%.4f" % MAP)
