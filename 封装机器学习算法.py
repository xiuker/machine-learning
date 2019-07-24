#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from ML.tree import DecisionTreeClassfier


# In[2]:


X = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')


# In[3]:


dt_clf = DecisionTreeClassfier()


# In[4]:


dt_clf.fit(X, y)


# In[5]:


dt_clf.tree_


# In[6]:


dt_clf.predict(X[:3])


# In[7]:


y_predict = dt_clf.predict(X)


# In[8]:


from ML.metrics import accuracy_score


# In[9]:


accuracy_score(y, y_predict)


# In[10]:


dt_clf.score(X, y)


# In[11]:


from ML.knn import KNeighborsClassifier


# In[12]:


knn_clf = KNeighborsClassifier()


# In[13]:


knn_clf.fit(X, y)


# In[14]:


knn_clf.predict(X[:3])


# In[15]:


knn_clf.score(X, y)


# In[ ]:




