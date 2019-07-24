#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


X = np.loadtxt('x.txt')


# In[3]:


y = np.loadtxt('y.txt')


# In[4]:


np.random.permutation(10)


# In[5]:


np.random.seed(1)
shuffle = np.random.permutation(len(y))


# In[6]:


test_size = 0.25
size = int(len(X) * test_size)
size


# In[7]:


test_index = shuffle[:size]
train_index = shuffle[size:]


# In[8]:


test_index


# In[9]:


train_index


# In[10]:


y[test_index]


# In[11]:


X_test = X[test_index]
y_test = y[test_index]
X_train = X[train_index]
y_train = y[train_index]


# In[12]:


from ML.knn import kNN_classify


# In[13]:


y_predict = kNN_classify(X_train, y_train, X_test)


# In[14]:


np.array(y_predict) == y_test


# In[15]:


sum(np.array(y_predict) == y_test)


# In[16]:


len(X_test)


# In[17]:


sum(np.array(y_predict) == y_test) / len(X_test)


# In[19]:


from ML.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed = 1)


# In[21]:


y_predict = kNN_classify(X_train, y_train, X_test)


# In[22]:


sum(np.array(y_predict) == y_test) / len(X_test)


# In[ ]:




