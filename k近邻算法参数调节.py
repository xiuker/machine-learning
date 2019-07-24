#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


X = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')


# In[3]:


from ML.model_selection import train_test_split


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=500)


# In[5]:


from ML.knn import kNN_classify


# In[6]:


y_predict = kNN_classify(X_train, y_train, X_test)


# In[7]:


from ML.metrics import accuracy_score


# In[8]:


accuracy_score(y_test, y_predict)


# In[11]:


best_score = 0
best_k = 0
for k in range(1,21):
    y_predict = kNN_classify(X_train, y_train, X_test, k=k)
    score = accuracy_score(y_test, y_predict)
    if score > best_score:
        best_score = score
        best_k = k
print('best_score:', best_score)
print('best_k:', best_k)
    
    


# In[12]:


# 网格搜索进行参数调节
best_score = 0
best_k = 0
best_p = 0
for k in range(1,21):
    for p in range(1,10):
        y_predict = kNN_classify(X_train, y_train, X_test, k=k, p=p)
        score = accuracy_score(y_test, y_predict)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = p
print('best_score:', best_score)
print('best_k:', best_k)
print('best_p:', best_p)


# In[ ]:




