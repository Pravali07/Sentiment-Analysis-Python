from twython import Twython
from nltk import PorterStemmer
import pandas as pd
import json
import re
import numpy as np 
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

credentials = {}
credentials['CONSUMER_KEY'] = 'XXXX'
credentials['CONSUMER_SECRET'] = 'XXXX'
credentials['ACCESS_TOKEN'] = 'XXXX'
credentials['ACCESS_SECRET'] = 'XXXX'

#access twitter api
python_tweets = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'], credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])

train = pd.read_csv('training_data_set.csv')


def get_text(id1):
    try:
        te = python_tweets.show_status(id = id1, tweet_mode = 'extended')  #get tweet text from tweet id
        r = re.findall("@[\w]*", te['full_text'])                          #removing @user mentions from tweets (because they are of no help)
        ur = re.findall("(?P<url>https?://[^\s]+)", te['full_text'])       #removing any links present in text
        text = te['full_text']
        for i in r:
            text = re.sub(i, "", text)
        for i in ur:
            text = re.sub(i, "", text)
        return text
    except:
        return '0'

train['tweet_text'] = np.vectorize(get_text)(train['tweet_id'])           #create a column in csv file with preprocessed tweet text
train['tweet_text'] = train['tweet_text'].str.replace("[^a-zA-Z#]", " ")  #replace special characters
train['tweet_text'] = train['tweet_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #remove short words

#stemming
tokens = train['tweet_text'].apply(lambda x: x.split())
ps = PorterStemmer()
tokens = tokens.apply(lambda x: [ps.stem(i) for i in x])
for i in range(len(tokens)):
    tokens[i] = ' '.join(tokens[i])
train['tweet_text'] = tokens

train.to_csv('training_data_set.csv', index=False)

