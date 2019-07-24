
def accuracy_score(y, y_predict):
    assert y.shape[0] == y_predict.shape[0]
    return sum(y == y_predict) / len(y)