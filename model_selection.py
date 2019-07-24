import numpy as np 

def train_test_split(X, y, test_size=0.25, seed=None):

    assert X.shape[0] == y.shape[0], 'X中的样本数量需要等于y中的标签数量'

    assert 0 <= test_size <= 1, 'test_size需要在0到1之间'

    if seed:
        np.random.seed(seed)

    shuffle = np.random.permutation(len(X))

    size = int(len(X) * test_size)

    test_index = shuffle[:size]
    train_index = shuffle[size:]

    X_test = X[test_index]
    y_test = y[test_index]
    X_train = X[train_index]
    y_train = y[train_index]

    return X_train, X_test, y_train, y_test