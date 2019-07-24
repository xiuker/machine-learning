import numpy as np
from collections import Counter
from ML.metrics import accuracy_score

class KNeighborsClassifier():
    """k近邻算法的类
    使用方法：实例化类
             调用fit函数
             调用predict函数得到预测后的特征值
             调用score函数得到预测的准确度"""
    def __init__(self, k=5, p=2):
        # 初始化类，k是k近邻的范围，默认为5
        # p是求距离的哪一种，1求曼哈顿距离，2求欧拉距离，>2求明可夫斯基距离，默认为2
        assert k > 0, 'k需要大于0'
        
        assert p > 0, 'p需要大于0'

        self.k, self.p = k, p
        self._X_train, self._y_train = None, None

    def fit(self, X_train, y_train):
        '''拟合函数，初始化X，y的训练数据'''
        assert X_train.shape[0] == y_train.shape[0], \
        'X_train中样本数量需要与y_train的数量相同'
        assert self.k <= y_train.shape[0], 'k需要小于或等于总的样本数'

        self._X_train, self._y_train = X_train, y_train

    def predict(self, X_predict):
        '''kNN分类器'''
        assert self._X_train.shape[1] == X_predict.shape[1], \
            '预测的特征数量需要等于样本的特征数量'
        return np.array([self._predict(x) for x in X_predict])

    def _predict(self, x):
        '''计算每个测试数据与训练数据的距离，返回一个特征值的矩阵'''
        # 列表表达式求得每个测试数据与训练数据的距离
        distances = [self._distance(item,x, p=self.p) for item in self._X_train]
        # argsort函数返回排序好后的索引，由小到大排序，取出前k个值
        nearest = np.argsort(distances)[:self.k]
        # 通过索引取出特征值
        k_labels = self._y_train[nearest]

        return Counter(k_labels).most_common(1)[0][0]

    def score(self, X_test, y_test):
        '''计算预测的准确度'''
        y_predict = self.predict(X_test)
        return self._accuracy_score(y_test, y_predict)

    def _accuracy_score(y, y_predict):
        '''准确度计算，输入正确的特征和预测的特征'''
        assert y.shape[0] == y_predict.shape[0]
        # y == y_predict 得到一个各位True或者False的矩阵，求得其中True的数量除以y的数量就是准确度
        return sum(y == y_predict) / len(y)

    def _distance(self, a, b, p):
        '''计算两个数据的距离，适用于欧拉距离，曼哈顿距离，明可夫斯基距离也适用于多维度'''
        return np.sum(np.abs(a - b) ** p) ** (1 / p)

def kNN_classify(X_train, y_train, X_predict, k=5, p=2):
    '''kNN分类器'''
    assert k > 0, 'k需要大于0'
    assert k <= y_train.shape[0], 'k需要小于或等于总的样本数'

    assert p > 0, 'p需要大于0'

    assert X_train.shape[0] == y_train.shape[0], \
        'X_train中样本数量需要与y_train的数量相同'
    assert X_train.shape[1] == X_predict.shape[1], \
        '预测的特征数量需要等于样本的特征数量'
    return np.array([_predict(X_train, y_train, x, k, p) for x in X_predict])

def _predict(X_train, y_train, x, k=5, p=2):
    distances = [distance(item,x, p=p) for item in X_train]
    nearest = np.argsort(distances)[:k]
    k_labels = y_train[nearest]

    return Counter(k_labels).most_common(1)[0][0]


def distance(a, b, p=2):
    '''计算两个数据的距离，适用于欧拉距离，曼哈顿距离，明可夫斯基距离
    也适用于多维度'''
    return np.sum(np.abs(a - b) ** p) ** (1 / p)