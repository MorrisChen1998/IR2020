data = ['s1','s2','s3']
for x in range(3):
    f = open(data[x]+'.txt')
    query = int(f.readline())
    MAP = 0
    for i in range(query):
        q_r = f.readline().split()
        q_a = f.readline().split()
        hit = 0
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
MAP = 0
for i in range(query):
    retrieved = input().split()
    answer = input().split()
    hit = 0
    for d in range(len(retrieved)):
        if(retrieved[d] in answer):
            hit += 1
            MAP += hit/((d+1)*len(answer)*query)
        if(hit >= len(answer)):
            break
print(round(MAP,4))