import numpy as np 
from collections import Counter


def calc(X_temp, features):
    res = {}
    for i in range(X_temp.shape[1]):
        feature_counter = Counter(X_temp[:,i])
        temp = {}
        for it in features[i]:
            temp[it] = feature_counter.get(it,0) / len(X_temp)
        res[i] = temp
    return res

class PusuClassifier(object):
    """docstring for PusuClassifier"""
    def __init__(self):
        self.features_probability = None
        self.label_probability = None
    def fit(self, X, y):
        self.label_probability = {}
        label = Counter(y)
        for i in label.keys():
            self.label_probability[i] = label[i] / len(y)

        features = {}
        for i in range(X.shape[1]):
            features[i] = np.unique(X[:,i])
        self.features_probability = {}
        for i in self.label_probability.keys():
            self.features_probability[i] = calc(X[y==i], features)

        return self

    def predict(self,x):
        assert self.features_probability is not None,'请先使用fit()方法'
        x_feature = {}
        for i in range(x.shape[0]):
            x_feature[i] = x[i]
        c = {}
        for j in self.features_probability.keys():
            a = self.label_probability[j]
            b = 1
            for k in x_feature.keys():
                b *= self.features_probability[j][k][x_feature[k]]
            a *= b
            c[j] = a
        return Counter(c).most_common(1)