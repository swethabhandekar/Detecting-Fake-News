#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


#default theme
plt.style.use('ggplot')
sns.color_palette("tab10")
sns.set(context='notebook', style='darkgrid', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[20,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


# In[3]:



df=pd.read_csv('C:/Users/sutra/OneDrive/Desktop/DEv-Intern/Python-Projects-Detecting-Fake-News/Data/news.csv')


# In[4]:


print(df.shape)


# In[5]:


df.head()


# In[6]:


labels=df.label


# In[7]:


labels.head()


# In[8]:


target=df.label.value_counts()
target


# In[9]:


sns.countplot(df.label)
plt.title('the number of news fake/real');


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[11]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[12]:


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 


# In[13]:


tfidf_test=tfidf_vectorizer.transform(x_test)


# In[15]:


pac=PassiveAggressiveClassifier(max_iter=50)


# In[16]:


pac.fit(tfidf_train,y_train)


# In[17]:


y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[18]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




