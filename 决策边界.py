#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


np.arange(1, 2, 0.2)


# In[3]:


np.arange(2, 3, 0.2)


# In[4]:


x, y = np.meshgrid(np.arange(1, 2, 0.2), np.arange(2, 3, 0.2))


# In[5]:


x


# In[6]:


y


# In[7]:


plt.scatter(x, y)
plt.show()


# In[8]:


X = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
X


# In[9]:


def height(x):
    # 1 2
    return np.abs(x[0] -1 ) * (x[1] - 2)


# In[10]:


z = np.array([height(x) for x in X])


# In[11]:


z


# In[12]:


z = z.reshape(x.shape)


# In[13]:


z


# In[14]:


# 绘制等高线
plt.contour(x, y, z)
plt.show()


# In[15]:


plt.contourf(x, y, z, 20, cmap=plt.cm.hot)
plt.show()


# In[16]:


X = np.loadtxt('x.txt')
X = X[:, 2:]
y = np.loadtxt('y.txt')


# In[17]:


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# In[22]:


def plot_decision_boundary(X, y, predict_func, step=0.1):
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    labels = predict_func(np.concatenate([x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)], axis=1))
    z = labels.reshape(x_mesh.shape)
    
    plt.contourf(x_mesh, y_mesh, z, cmap=plt.cm.Spectral)
    
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


# In[23]:


from ML.knn import KNeighborsClassifier


# In[24]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X, y)


# In[25]:


plot_decision_boundary(X, y, lambda x: knn_clf.predict(x))


# In[26]:


from ML.tree import DecisionTreeClassfier


# In[27]:


dt_clf = DecisionTreeClassfier()
dt_clf.fit(X, y)


# In[30]:


get_ipython().run_cell_magic('time', '', 'plot_decision_boundary(X, y, lambda x: knn_clf.predict(x), step=0.01)')


# In[31]:


get_ipython().run_cell_magic('time', '', 'plot_decision_boundary(X, y, lambda x: dt_clf.predict(x), step=0.01)')


# In[ ]:




