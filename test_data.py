#import packages
from twython import Twython
from nltk import PorterStemmer
import pandas as pd
import json
import re
import numpy as np 
import warnings
import csv
warnings.filterwarnings("ignore", category=DeprecationWarning)

#connect to twitter account
credentials = {}
credentials['CONSUMER_KEY'] = 'XXXXX'
credentials['CONSUMER_SECRET'] = 'XXXXXXX'
credentials['ACCESS_TOKEN'] = 'XXXXXXX'
credentials['ACCESS_SECRET'] = 'XXXXXX'

python_tweets = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'], credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])


#tweets containing the following hashtags
corona = {'q': '#corona',
        'result_type': 'popular',
        'count': 10,
        'lang': 'en',
        }
lockdown = {'q': '#lockdown',
        'result_type': 'popular',
        'count': 10,
        'lang': 'en',
        }
covid = {'q': '#covid19',
        'result_type': 'popular',
        'count': 10,
        'lang': 'en',
        }
corona_virus = {'q': '#coronavirus',
        'result_type': 'popular',
        'count': 10,
        'lang': 'en',
        }
locEx = {'q': '#lockdownextension',
        'result_type': 'popular',
        'count': 10,
        'lang': 'en',
        }

#preprocessing the data
def get_text(id1):
    te = python_tweets.show_status(id = id1, tweet_mode = 'extended')
    #print(te)
    r = re.findall("@[\w]*", te['full_text'])
    ur = re.findall("(?P<url>https?://[^\s]+)", te['full_text'])
    text = te['full_text']
    for i in r:
        text = re.sub(i, "", text)
    for i in ur:
        text = re.sub(i, "", text)
    return text

details = {'user': [], 'date': [], 'id': [], 'favorite_count': []}
count=0
for status in python_tweets.search(**corona)['statuses']:
    count+=1
    details['user'].append(status['user']['screen_name'])
    details['date'].append(status['created_at'])
    details['id'].append(status['id'])
    details['favorite_count'].append(status['favorite_count'])
for status in python_tweets.search(**corona_virus)['statuses']:
    count+=1
    details['user'].append(status['user']['screen_name'])
    details['date'].append(status['created_at'])
    details['id'].append(status['id'])
    details['favorite_count'].append(status['favorite_count'])
for status in python_tweets.search(**lockdown)['statuses']:
    count+=1
    details['user'].append(status['user']['screen_name'])
    details['date'].append(status['created_at'])
    details['id'].append(status['id'])
    details['favorite_count'].append(status['favorite_count'])
for status in python_tweets.search(**covid)['statuses']:
    count+=1
    details['user'].append(status['user']['screen_name'])
    details['date'].append(status['created_at'])
    details['id'].append(status['id'])
    details['favorite_count'].append(status['favorite_count'])
for status in python_tweets.search(**locEx)['statuses']:
    count+=1
    details['user'].append(status['user']['screen_name'])
    details['date'].append(status['created_at'])
    details['id'].append(status['id'])
    details['favorite_count'].append(status['favorite_count'])
dict_data = []
for i in range(count):
    d = {'user': [], 'date': [], 'id': [], 'favorite_count': []}
    d['user'] = details['user'][i]
    d['date'] = details['date'][i]
    d['id'] = details['id'][i]
    d['favorite_count'] = details['favorite_count'][i]
    dict_data.append(d)

csv_columns = ['user','date','id', 'favorite_count']
csv_file = "TestDataSet.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")

def get_text(id1):
    te = python_tweets.show_status(id = id1, tweet_mode = 'extended')
    #print(te)
    r = re.findall("@[\w]*", te['full_text'])
    ur = re.findall("(?P<url>https?://[^\s]+)", te['full_text'])
    text = te['full_text']
    for i in r:
        text = re.sub(i, "", text)
    for i in ur:
        text = re.sub(i, "", text)
    return text

test = pd.read_csv('TestDataSet.csv')

test['tweet_text'] = np.vectorize(get_text)(test['id'])
test['tweet_text'] = test['tweet_text'].str.replace("[^a-zA-Z#]", " ")
test['tweet_text'] = test['tweet_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tokens = test['tweet_text'].apply(lambda x: x.split())
ps = PorterStemmer()
tokens = tokens.apply(lambda x: [ps.stem(i) for i in x])
for i in range(len(tokens)):
    tokens[i] = ' '.join(tokens[i])
test['tweet_text'] = tokens
test.to_csv('TestDataSet.csv', index = False)

    



