"""
Create two csv files to store positive and negative words along with their counts (no of occurrences).
"""

import pandas as pd
import nltk
import heapq
import numpy as np
import re
import csv

data = pd.read_csv('training_data_set.csv')

posCount = {}
negCount = {}
cp = 0
cn = 0
for i in range(len(data)):
    words = nltk.word_tokenize(data['tweet_text'][i])
    if data['sentiment'][i] > 0:
        for word in words:
            cp += 1
            if word not in posCount.keys():
                posCount[word] = 1
            else:
                posCount[word] += 1
    else:
        for word in words:
            cn += 1
            if word not in negCount.keys():
                negCount[word] = 1
            else:
                negCount[word] += 1
                
wp = list(posCount.keys())
dict_datap = []
dict_datan = []
for i in wp:
    d = {'words':[], 'count':[], 'sentiment':[]}
    d['words'] = i
    d['count'] = posCount[i]
    d['sentiment'] = 1
    dict_datap.append(d)

wn = list(negCount.keys())
for i in wn:
    d = {'words':[], 'count':[], 'sentiment':[]}
    d['words'] = i
    d['count'] = negCount[i]
    d['sentiment'] = 0
    dict_datan.append(d)

csv_columns = ['words','count','sentiment']
try:
    with open("PosWordsCount.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_datap:
            writer.writerow(data)
except IOError:
    print("I/O error")

try:
    with open("NegWordsCount.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_datan:
            writer.writerow(data)
except IOError:
    print("I/O error")







    
