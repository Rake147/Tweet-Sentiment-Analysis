#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
import re
import nltk 


# In[2]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/twitter.csv')


# In[3]:


data.head()


# In[4]:


nltk.download('stopwords')
stemmer=nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


# In[5]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[6]:


data['tweet']=data['tweet'].apply(clean)


# In[7]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data['Positive']=[sentiments.polarity_scores(i)['pos'] for i in data['tweet']]
data['Negative']=[sentiments.polarity_scores(i)['neg'] for i in data['tweet']]
data['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['tweet']]


# In[9]:


data=data[['tweet','Positive','Negative','Neutral']]


# In[10]:


data.head()


# In[11]:


x=sum(data['Positive'])
y=sum(data['Negative'])
z=sum(data['Neutral'])

def sentiment_score(a,b,c):
    if (a>b) and (a>c):
        print("Positive")
    elif (b>a) and (b>c):
        print('Negative')
    else:
        print('Neutral')

        


# In[12]:


sentiment_score(x,y,z)


# In[13]:


print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)

