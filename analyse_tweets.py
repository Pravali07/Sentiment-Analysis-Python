#import packages
from twython import Twython
from nltk import PorterStemmer
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings("ignore",category = DeprecationWarning)

pos_word = pd.read_csv('PosWordsCount.csv')
neg_word = pd.read_csv('NegWordsCount.csv')
dataset = pd.read_csv('TestDataSet.csv')


#Naive Bayes
def get_probability(s):
    sw = nltk.word_tokenize(s)
    words_pos = list(pos_word['words'])
    count_pos = list(pos_word['count'])
    words_neg = list(neg_word['words'])
    count_neg = list(neg_word['count'])

    prob = len(words_pos)/len(words_neg)

    for i in sw:
        p1, n1 = 0,0
        for j in range(len(words_pos)):
            if (words_pos[j] == i):
                p1 += count_pos[j]
                break
        for j in range(len(words_neg)):
            if (words_neg[j] == i):
                n1 += count_neg[j]
                break
        if p1 == 0:
            continue
        prob = prob*(p1/(p1+n1))

    prob1 = len(words_neg)/len(words_pos)

    for i in sw:
        p1,n1 = 0,0
        for j in range(len(words_pos)):
            if (words_pos[j] == i):
                p1 += count_pos[j]
                break
        for j in range(len(words_neg)):
            if (words_neg[j] == i):
                n1 += count_neg[j]
                break
        if n1 == 0:
            continue
        prob1 = prob1*(n1/(p1+n1))

    if prob > prob1:
        return prob/(prob+prob1)
    else:
        return prob1/(prob+prob1)

dataset['sentiment'] = np.vectorize(get_probability)(dataset['tweet_text'])
dataset.to_csv('TestDataSet.csv',index = False)
po = 0
no = 0



#plotting data
for p in dataset['sentiment']:
    if p > 0.55:
        po += 1
    else:
        no += 1
objects = ('Positive','Negative')
y_pos = np.arange(len(objects))
performance = [po,no]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.xlabel('Sentiment')
plt.title('Sentiment Analysis')

plt.show()
        
        
                
    
