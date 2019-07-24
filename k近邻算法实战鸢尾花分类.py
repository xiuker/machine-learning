#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_excel('./03-发给学员/iris.xlsx')
df.head()


# In[3]:


X_train = df.values[:,:4]
X_train.shape
X_train


# In[4]:


y_train = df.values[:,-1]
y_train.shape


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.scatter(X_train[y_train==0, 0],X_train[y_train==0, 1], color='r')
plt.scatter(X_train[y_train==1, 0],X_train[y_train==1, 1], color='g')
plt.scatter(X_train[y_train==2, 0],X_train[y_train==2, 1], color='b')
plt.show()


# In[7]:


plt.scatter(X_train[y_train==0, 2],X_train[y_train==0, 3], color='r')
plt.scatter(X_train[y_train==1, 2],X_train[y_train==1, 3], color='g')
plt.scatter(X_train[y_train==2, 2],X_train[y_train==2, 3], color='b')
plt.show()


# In[8]:


from ML.knn import kNN_classify


# In[9]:


X_predict = np.array([[4.9, 3. , 1.4, 0.2],[5.9, 3. , 5.1, 1.8]])
kNN_classify(X_train, y_train, X_predict)


# In[15]:


np.savetxt('x.txt', X_train)


# In[16]:


np.savetxt('y.txt', y_train)


# In[ ]:




