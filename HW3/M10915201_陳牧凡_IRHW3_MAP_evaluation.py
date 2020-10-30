# -*- coding: utf-8 -*-
# M10915201_陳牧凡_IRHW3
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
print(round(MAP, 4))
