#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from ML.knn import kNN_classify
from ML.model_selection import train_test_split
from ML.metrics import accuracy_score


# In[15]:


# 有两种品种的树苗
# 树苗的直径cm  生长天数
X = np.array([
    [1.0, 100],
    [1.1, 200],
    [0.9, 150],
    [0.2, 190],
    [1.0, 100],
    [1.1, 200],
    [0.7, 150],
    [0.2, 190],
    [2.1, 250],
    [1.8, 220],
    [2.2, 290],
    [1.9, 270],
    [2.1, 390],
    [1.8, 220],
    [2.2, 258],
    [1.9, 360],
])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


# In[16]:


plt.scatter(X[y==0, 0], X[y==0, 1], color='r')
plt.scatter(X[y==1, 0], X[y==1, 1], color='g')
plt.show()


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed = 100)
y_predict = kNN_classify(X_train, y_train, X_test)
accuracy_score(y_test,y_predict)


# In[9]:


# 标准化按照比例进行缩放，使其落入小的空间之中
# 变成均值为0，标准差为1的数据,去除量纲对结果的影响


# In[18]:


X[:, 0] = (X[:, 0] - np.mean(X[:, 0])) / np.std(X[:, 0])
X[:, 0]


# In[19]:


np.mean(X[:, 0])


# In[20]:


np.std(X[:, 0])


# In[23]:


X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])
X[:, 1]


# In[25]:


X


# In[27]:


y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


# In[28]:


plt.scatter(X[y==0, 0], X[y==0, 1], color='r')
plt.scatter(X[y==1, 0], X[y==1, 1], color='g')
plt.show()


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed = 100)
y_predict = kNN_classify(X_train, y_train, X_test)
accuracy_score(y_test,y_predict)


# In[30]:


from ML.preprocessing import StandardScaler


# In[31]:


# 有两种品种的树苗
# 树苗的直径cm  生长天数
X = np.array([
    [1.0, 100],
    [1.1, 200],
    [0.9, 150],
    [0.2, 190],
    [1.0, 100],
    [1.1, 200],
    [0.7, 150],
    [0.2, 190],
    [2.1, 250],
    [1.8, 220],
    [2.2, 290],
    [1.9, 270],
    [2.1, 390],
    [1.8, 220],
    [2.2, 258],
    [1.9, 360],
])


# In[33]:


standardScaler = StandardScaler()
standardScaler.fit(X)


# In[35]:


standardScaler.mean_


# In[36]:


standardScaler.scale_


# In[39]:


X2 = standardScaler.transform(X)
X2


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.25, seed = 100)
y_predict = kNN_classify(X_train, y_train, X_test)
accuracy_score(y_test,y_predict)


# In[ ]:




