import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

# 训练数据集
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 可视化
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], color='g')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], color='r')
plt.show()

# 新数据
x = np.array([8.093607318, 3.365731514])

# 可视化
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], color='g')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], color='r')
plt.scatter(x[0], x[1], color='b')
plt.show()

# kNN的过程
distances = []
for x_train in X_train:
    d = sqrt(np.sum((x_train - x) ** 2))
    distances.append(d)

# 或者
distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]

# 返回索引
nearest = np.argsort(distances)

k = 6
topK_y = [y_train[i] for i in nearest[:k]]

votes = Counter(topK_y)

predict_y = votes.most_common(1)[0][0]

# 最终结果
predict_y

# 封装kNN算法
def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0],\
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X—train"
    
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]

predict_y = kNN_classify(6, X_train, y_train, x)

# ------------------------------- #
# 使用scikit-learn中的kNN
# ------------------------------- #
from sklearn.neighbors import KNeighborsClassifier

# 新数据
x = np.array([8.093607318, 3.365731514])

kNN_classifier = KNeighborsClassifier(n_neighbors=6)    # 创建实例，设置k
kNN_classifier.fit(X_train, y_train)    # 拟合训练数据集
kNN_classifier.predict(x.reshape(1,-1)) # 哪怕只是预测一个值，也要写成矩阵形式

# 用类改写scikit-learn中的kNN
class KNNClassifier:
    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >=1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k"
        
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] ==self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        
        return votes.most_common(1)[0][0]
    
    def score(self, X_test, y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k

# 测试类是否生效
X_predict = x.reshape(1, -1)
knn_clf = KNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_predict)


# ------------------------------- #
# 训练集与测试集
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 加载数据集
iris = datasets.load_iris()

X = iris.data
y = iris.target

# train_test_split
# 1. 索引乱序化处理(一一对应)
shuffle_indexes = np.random.permutation(len(X))
# 2. 指定训练数据集
test_ratio = 0.2    # 测试集占比
test_size = int(len(X) * test_ratio)
test_indexes = shuffle_indexes[:test_size]  # 前20%测试训练集
train_indexes = shuffle_indexes[test_size:] # 后80%训练数据集

X_train = X[train_indexes]
y_train = y[train_indexes]
X_test = X[test_indexes]
y_test = y[test_indexes]

# 以上过程封装成函数
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

# sklearn中的train_test_split
from sklearn.model_selection import train_test_split

# 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)


# ------------------------------- #
# 分类准确度
# ------------------------------- #
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
digits = datasets.load_digits()
digits.keys()
print(digits.DESCR)

X = digits.data
X.shape
y = digits.target
y.shape
digits.target_names
y[:100]
X[:10]

# 查看某一个值
some_digit = X[666]
y[666]
some_digit_image = some_digit.reshape(8,8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()

# 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

my_knn_clf = KNeighborsClassifier(n_neighbors=3)    # 创建实例，设置k
my_knn_clf.fit(X_train, y_train)    # 拟合训练数据集
y_predict = my_knn_clf.predict(X_test)  # 预测

# 比对准确度
sum(y_predict == y_test) / len(y_test)

# 函数封装 准确率
def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    
    return sum(y_true == y_predict) / len(y_true)

# 检验函数是否正确
accuracy_score(y_test, y_predict)

# scikit-learn中的accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.predict(X_test)

accuracy_score(y_test, y_predict)
# 或者
knn_clf.score(X_test, y_test)

# ------------------------------- #
# 超参数：对比模型参数
# ------------------------------- #

# 超参数：在算法运行前需要决定的参数
# 模型参数：算法过程中学习的参数
# 寻找好的超参数的方法：1. 领域知识 2. 经验数值 3. 实验搜索

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载数据
digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

# 寻找最好的k
best_score = 0.0
best_k = -1
for k in range(1,11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test,y_test)
    if score > best_score:
        best_k = k
        best_score = score

print("best_k = ", best_k)
print("best_score = ", best_score)

# 考虑距离的情况?
best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"]:
    for k in range(1,11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test,y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method

print("best_method = ", best_method)
print("best_k = ", best_k)
print("best_score = ", best_score)

# 关于距离的定义
# 1. 欧拉距离 2. 曼哈顿距离 3. 明可夫斯基距离（超参数p）

# 搜索明可夫斯基距离相应的p
import timeit

best_p = -1
best_score = 0.0
best_k = -1

start = timeit.default_timer()

for k in range(1,11):
    for p in range(1,6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test,y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_p = p

end = timeit.default_timer()

print("best_p = ", best_p)
print("best_k = ", best_k)
print("best_score = ", best_score)
print("run time: ", round(end-start,4),"s")


# ------------------------------- #
# 网格搜索与k近邻算法中更多的超参数
# ------------------------------- #

# Grid Search
from sklearn.model_selection import GridSearchCV    # CV:交叉验证
from sklearn.neighbors import KNeighborsClassifier
# 定义
para_grid = [
    {
        'weights' : ['uniform'],
        'n_neighbors' : [i for i in range(1,11)]
    },
    {
        'weights' : ['distance'],
        'n_neighbors' : [i for i in range(1,11)],
        'p' : [i for i in range(1,6)]
    }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, para_grid)
grid_search.fit(X_train, y_train)

grid_search.best_estimator_
grid_search.best_score_
grid_search.best_params_

knn_clf = grid_search.best_estimator_
knn_clf.predict(X_test)
knn_clf.score(X_test, y_test)

grid_search = GridSearchCV(knn_clf, para_grid, n_jobs=-1, verbose=2)    # n_jobs: n核处理数据.n=-1时全负荷运算; 
                                                                        # verbose: 搜索过程输出
grid_search.fit(X_train, y_train)


# ------------------------------- #
# 数据归一化Feature Scaling
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0,100,100)

# 最值归一化
(x - np.min(x)) / (np.max(x) - np.min(x))
#对于矩阵
X = np.random.randint(0, 100, (50,2))
X = np.array(X, dtype=float)
X[:, 0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
X[:, 1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))

plt.scatter(X[:,0], X[:,1])
plt.show()

# 均值方差归一化 Standardization
X2 = np.random.randint(0,100,(50,2))
X2 = np.array(X2, dtype=float)
X2[:,0] = (X2[:,0] - np.mean(X2[:,0])) / np.std(X2[:,0])
X2[:,1] = (X2[:,1] - np.mean(X2[:,1])) / np.std(X2[:,1])

# ------------------------------- #
# scikit-learn中的Scaler
# ------------------------------- #
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 666)

standardScaler = StandardScaler()
standardScaler.fit(X_train)
standardScaler.mean_    #均值
standardScaler.scale_   # 标准差

X_train = standardScaler.transform(X_train)   # 归一化
X_test_standard = standardScaler.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test_standard,y_test)   # 需传入归一化后的数据

