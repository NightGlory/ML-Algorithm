# ------------------------------- #
# 决策树Decision Tree
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,2:]
y = iris.target

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_clf.fit(X,y)

# 不规则的决策边界绘制方法
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100))
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

# 总结：决策树：
# 1. 非参数学习算法
# 2. 可以解决分类问题，天然可以解决多分类问题
# 3. 也可以解决回归问题
# 4. 可解释性


# ------------------------------- #
# 信息熵:以二分类为例
# ------------------------------- #
def entropy(p):
    return -p*np.log(p) - (1-p)*np.log(1-p)

x = np.linspace(0.01,0.99,200)
plt.plot(x, entropy(x))
plt.show()

# 使用信息熵寻找最优化分
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_clf.fit(X,y)

def split(X, y, d, value):
    index_a = (X[:,d] <= value)
    index_b = (X[:,d] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]

# 计算熵
from collections import Counter
from math import log

def entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p)
    return res

def try_split(X, y):
    best_entropy = float('inf')
    best_d, best_v = -1, -1
    for d in range(X.shape[1]):
        sort_index = np.argsort(X[:,d])
        for i in range(1, len(X)):
            if X[sort_index[i-1], d] != X[sort_index[i], d]:
                v = (X[sort_index[i-1], d] + X[sort_index[i], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                e = entropy(y_l) + entropy(y_r)
                if e < best_entropy:
                    best_entropy, best_d, best_v = e, d, v
    
    return best_entropy, best_d, best_v

# 函数调用
best_entropy, best_d, best_v = try_split(X,y)
print("best entropy = ", best_entropy)
print("best d = ", best_d)
print("best v = ", best_v)

# 查看参数
X1_l, X1_r, y1_l, y1_r = split(X, y, best_d, best_v)
entropy(y1_l)
entropy(y1_r)

# 第二个划分
best_entropy2, best_d2, best_v2 = try_split(X1_r, y1_r)
print("best entropy = ", best_entropy2)
print("best d = ", best_d2)
print("best v = ", best_v2)

X2_l, X2_r, y2_l, y2_r = split(X1_r, y1_r, best_d2, best_v2)
entropy(y2_l)
entropy(y2_r)


# ------------------------------- #
# 基尼系数
# ------------------------------- #
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=2, criterion="gini")
dt_clf.fit(X, y)

# 绘制
plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

# 模拟使用基尼系数划分
from collections import Counter
from math import log

# 计算基尼系数
def gini(y):
    counter = Counter(y)
    res = 1.0
    for num in counter.values():
        p = num / len(y)
        res -= p**2
    return res

def try_split(X, y):
    best_g = float('inf')
    best_d, best_v = -1, -1
    for d in range(X.shape[1]):
        sort_index = np.argsort(X[:,d])
        for i in range(1, len(X)):
            if X[sort_index[i-1], d] != X[sort_index[i], d]:
                v = (X[sort_index[i-1], d] + X[sort_index[i], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                g = gini(y_l) + gini(y_r)
                if g < best_g:
                    best_g, best_d, best_v = g, d, v
    
    return best_g, best_d, best_v

# 函数调用
best_g, best_d, best_v = try_split(X,y)
print("best gini = ", best_g)
print("best d = ", best_d)
print("best v = ", best_v)

# 第二次划分
best_g2, best_d2, best_v2 = try_split(X1_r, y1_r)
print("best gini = ", best_g2)
print("best d = ", best_d2)
print("best v = ", best_v2)


# ------------------------------- #
# CART与决策树中的超参数
# ------------------------------- #
# 剪枝：降低复杂度，解决过拟合
from sklearn import datasets
X, y = datasets.make_moons(noise=0.25, random_state=666)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)

plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
# 过拟合

# 调参
dt_clf2 = DecisionTreeClassifier(max_depth=2)
dt_clf2.fit(X,y)
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 调参
dt_clf3 = DecisionTreeClassifier(min_samples_split=10)
dt_clf3.fit(X,y)
plot_decision_boundary(dt_clf3, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 调参
dt_clf4 = DecisionTreeClassifier(min_samples_leaf=6)
dt_clf4.fit(X,y)
plot_decision_boundary(dt_clf4, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 调参
dt_clf5 = DecisionTreeClassifier(max_leaf_nodes=4)
dt_clf5.fit(X,y)
plot_decision_boundary(dt_clf5, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()


# ------------------------------- #
# 决策树解决回归问题
# ------------------------------- #
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
dt_reg.score(X_test, y_test)
# 对比: 过拟合
dt_reg.score(X_train, y_train)