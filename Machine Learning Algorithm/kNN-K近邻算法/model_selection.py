import numpy as np

# sklearn中的train_test_split封装成函数
def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X和y按照test_radio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <=1.0, \
        "test_radio must be valid"

    if seed:
        np.random.seed(seed)
    
    shuffle_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]  # 前20%测试训练集
    train_indexes = shuffle_indexes[test_size:] # 后80%训练数据集
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]