# -*- coding: utf-8 -*-

'''
2
d123 d84 d56 d6 d8 d9 d511 d129 d187 d25 d38 d48 d250 d113 d3
d3 d123 d25 d56 d9
d84 d56 d123 d129 d8 d6 d511 d9 d187 d3 d48 d38 d25 d113 d250
d123 d3 d6
'''
query = int(input())
MAP = 0
for i in range(query):
    q_r = input().split(' ')
    q_a = input().split(' ')
    hit = 0
    for d in range(len(q_r)):
        if(q_r[d] in q_a):
            hit+=1
            MAP+=hit/((d+1)*len(q_a)*query)
            
print("%.4f" % MAP)