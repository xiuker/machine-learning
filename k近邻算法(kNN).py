#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


np.random.seed(1)
x1 = np.random.randint(1,10,size=10).reshape(-1,2) # 生成随机数
x1


# In[3]:


np.random.seed(1)
x2 = np.random.randint(10,20,size=10).reshape(-1,2)
x2


# In[4]:


X_train = np.concatenate([x1,x2]) # 合并两个样本


# In[5]:


y_train = np.array([0,0,0,0,0,1,1,1,1,1]) # 用0，1标记分组


# In[6]:


plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],color='r')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],color='g')
plt.show()


# In[7]:


x = np.array([9,10])


# In[8]:


plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='r')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='g')
plt.scatter(x[0], x[1], color='b')
plt.show()


# In[9]:



def distance(a,b,p=2):
    '''计算两个数据的距离，适用于欧拉距离，曼哈顿距离，明可夫斯基距离
也适用于多维度'''
    return np.sum(np.abs(a-b)**p)**(1/p)


# In[10]:


# 循环计算样本与x的距离，放入到列表中
distances = []
for item in X_train:
    distances.append(distance(x,item))
distances


# In[11]:


# 用列表解析式更好些
distances = [distance(item,x) for item in X_train]


# In[12]:


# np的argsort排序算法，算出排序好的的值索引
ind = np.argsort(distances)
ind


# In[13]:


distances[ind[0]]


# In[14]:


X_train[ind]


# In[15]:


X_train


# In[16]:


k = 3  # 定义k，


# In[17]:


X_train[ind[:k]]


# In[18]:


y_train[ind[:k]]


# In[19]:


# 导入计数，用于统计列表的各个数据的数量
from collections import Counter
votes = Counter(y_train[ind[:k]])
votes


# In[20]:


votes.most_common(1)  # 统计最多的数的数量


# In[21]:


votes.most_common(1)[0][0]


# In[22]:


predict_y = votes.most_common(1)[0][0]
predict_y # 0是样本的标记


# In[ ]:




