#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## 欧拉距离

# In[2]:


X = np.array([[5,2],  # a
              [1,4]]) # b
plt.plot(X[:,0],X[:,1],'r-o')
plt.show()


# In[3]:



plt.plot(X[:,0],X[:,1],'r-o')
# 描点ab
plt.annotate('a',xy=X[0])
plt.annotate('b',xy=X[1])
# 画线直角三角形
plt.plot([X[:,0].min(),X[0][0]],[X[:,1].min(),X[0][1]],'g--')
plt.plot([X[:,0].min(),X[1][0]],[X[:,1].min(),X[1][1]],'g--')
plt.show()


# In[5]:


((X[0][0]-X[1][0])**2+(X[0][1]-X[1][1])**2)**0.5


# In[6]:


X[0]-X[1]


# In[16]:


(np.sum((X[0]-X[1])**2))**0.5


# ## 曼哈顿距离

# In[17]:


def distance(a,b,p=2):
    return np.sum(np.abs(a-b)**p)**(1/p)
# 支持多维度的数据


# In[18]:


distance(X[0],X[1])


# In[19]:


distance(X[0],X[1],p=1) # p=1就是曼哈顿距离


# In[20]:


X = np.array([[5,2,3],  # a
              [1,4,6]]) # b


# In[21]:


distance(X[0],X[1])


# In[ ]:




