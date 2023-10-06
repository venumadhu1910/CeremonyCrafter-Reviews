#!/usr/bin/env python
# coding: utf-8

# # The wedding reviews and details 2020

# In[63]:


import pandas as pd


# In[64]:


df = pd.read_csv(r"C:\Users\ANESTHESIA\Downloads\wedding_reddit.csv")


# In[65]:


df.head()


# In[66]:


df.isnull().sum()


# In[67]:


df.columns


# In[68]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])


# In[69]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[70]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] 
    post_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[post_indices]


# In[71]:


recommended_posts = get_recommendations('Just got married today!')
print(recommended_posts)


# In[72]:


recommended_posts = get_recommendations('I got married today @ The Frick 🥰')
print(recommended_posts)


# In[73]:


recommended_posts = get_recommendations('I got married today! Write me a trigger event')
print(recommended_posts)


# In[74]:


recommended_posts = get_recommendations('Got married today!')
print(recommended_posts)


# In[75]:


recommended_posts = get_recommendations('RAY AND TINA GOT MARRIED TODAY!')
print(recommended_posts)


# In[76]:


recommended_posts = get_recommendations('Buddy got married today. Watched through zoom.')
print(recommended_posts)


# In[ ]:




