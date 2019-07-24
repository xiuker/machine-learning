import numpy as np
from collections import Counter
from ML.metrics import accuracy_score

def gini(y):
    '''基尼系数函数，等于1-（每个值数量除以总数量的平方的总和）
        基尼系数越小，表示纯度越高'''
    counter = Counter(y)
    result = 0
    # counter类似与{1：2，2：3}
    for v in counter.values():
        result += (v/ len(y))**2
    return 1-result

def cut(X, y, v, d):
    '''在维度d或者说在特征d列，通过分割点v，将数据分成左右两个部分'''
    ind_left = (X[:, d] <= v)
    ind_right = (X[:, d] > v)
    return X[ind_left], X[ind_right], y[ind_left], y[ind_right]

def try_split(X, y, min_sample_leaf):
    # 双循环得到在d维度，分割点v，得到最好的g基尼系数
    best_g, best_d, best_v = 1, -1, -1
    for d in range(X.shape[1]):
        # argsort返回由小到大排序后的索引序列
        sorted_ind = np.argsort(X[:, d])
        for i in range(len(X) - 1):
            # 如果相邻的两个值相等则跳过分割
            if X[sorted_ind[i], d] == X[sorted_ind[i+1], d]:
                continue
            # 分割点v=相邻两值的均值
            v = (X[sorted_ind[i], d] + X[sorted_ind[i+1], d])/2
            X_left, X_right, y_left, y_right = cut(X, y, v, d)
            g_all = gini(y_left) + gini(y_right)
            # 基尼系数越好，并且左右叶节点大于等于最小数量
            if g_all < best_g and len(y_left) >= min_sample_leaf and len(y_right) >= min_sample_leaf:
                best_g, best_d, best_v =g_all, d, v
    return best_d, best_v, best_g

class DecisionTreeClassfier():
    """决策树类，使用方法：
        实例化类；
        调用fit函数，拟合此实例；
        调用predict函数，返回预测的结果；
        调用score函数，返回预测的准确度"""
    def __init__(self, max_depth=2, min_sample_leaf=1):
        '''初始化最大深度值和节点最小数量
            最大深度值默认为2，节点最小数量为1'''
        self.tree_ = None
        self.max_depth = 2
        self.min_sample_leaf = 1
    def fit(self, X, y):
        '''拟合函数，得到决策树'''
        self.tree_ = self.create_tree(X, y)
        return self

    def create_tree(self, X, y, current_depth=1):
        '''制造决策树的函数，当前深度参数用于对比最大深度'''
        # 判断，如果为真，返回None，不再分支
        if current_depth > self.max_depth:
            return None
        # d是分割维度，v是分割点，g是基尼系数
        d, v, g = try_split(X, y, self.min_sample_leaf)

        if d == -1 or g == 0:
            return None
        # 实例化一个节点的类
        node = Node(d, v, g)
        # 分割左右节点的数据和特征
        X_left, X_right, y_left, y_right = cut(X, y, v, d)
        # 递归获得左节点树
        node.children_left = self.create_tree(X_left, y_left, current_depth+1)
        if node.children_left is None:
            # label 获得标签
            label = Counter(y_left).most_common(1)[0][0]
            node.children_left = Node(l=label)
        # 递归获得右节点的树
        node.children_right = self.create_tree(X_right, y_right, current_depth+1)
        if node.children_right is None:
            # label
            label = Counter(y_right).most_common(1)[0][0]
            node.children_right = Node(l=label)
            
        return node

    def predict(self, X):
        '''返回预测的结果的矩阵'''
        assert self.tree_ is not None,'请先调用fit()方法'

        return np.array([self._predict(x, self.tree_) for x in X])

    def _predict(self, x, node):
        '''一个预测数据的节点标签'''
        if node.label is not None:
            return node.label
        
        if x[node.dim] <= node.value:
            #left
            return self._predict(x, node.children_left)
        else:
            #right
            return self._predict(x, node.children_right)
    def score(self, X_test, y_test):
        '''计算预测的准确度'''
        y_predict = self.predict(X_test)
        return self._accuracy_score(y_test, y_predict)

    def _accuracy_score(self, y, y_predict):
        '''准确度计算，输入正确的特征和预测的特征'''
        assert y.shape[0] == y_predict.shape[0]
        # y == y_predict 得到一个各位True或者False的矩阵，求得其中True的数量除以y的数量就是准确度
        return sum(y == y_predict) / len(y)

class Node():
    '''节点类,d维度/特征列，v分割点，g基尼系数，l标签/特征值'''
    def __init__(self, d=None, v=None, g=None, l=None):
        self.dim, self.value, self.gini, self.label = d, v, g, l
        self.children_left, self.children_right = None, None
    def __repr__(self):
        return 'Node(d={}, v={}, g={}, l={})'.format(self.dim, self.value, self.gini, self.label)
