# ------------------------------- #
# 逻辑回归
# ------------------------------- #
# 解决分类问题：将样本的特征和样本发生的概率联系起来，概率是一个数(二元分类)

# Sigmoid函数
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(t):
    return 1/(2+np.exp(-t))

x = np.linspace(-10, 10, 500)
y = sigmoid(x)

plt.plot(x, y)
plt.show()


# ------------------------------- #
# 逻辑回归算法
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y<2, :2]
y = y[y<2]

X.shape

plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

# 使用逻辑回归
from sklearn.model_selection import train_test_split
X_trian, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from LogisticRegression import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_trian, y_train)
log_reg.score(X_test, y_test)

log_reg.predict_proba(X_test)
y_test

log_reg.predict(X_test)


# ------------------------------- #
# 决策边界
# ------------------------------- #
def x2(x1):
    return (-log_reg.coef_[0] * x1 - log_reg.intercept_) / log_reg.coef_[1]

x1_plot = np.linspace(4,8,1000)
x2_plot = x2(x1_plot)

plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.plot(x1_plot, x2_plot)
plt.show()

# 测试集决策边界
plt.scatter(X_test[y_test==0,0], X_test[y_test==0,1],color='r')
plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1],color='b')
plt.plot(x1_plot, x2_plot)
plt.show()

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

plot_decision_boundary(log_reg, axis=[4,7.5,1.5,4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# kNN决策边界
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_trian, y_train)

knn_clf.score(X_test, y_test)

plot_decision_boundary(knn_clf, axis=[4,7.5,1.5,4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# kNN三个分类
knn_clf_all = KNeighborsClassifier()
knn_clf_all.fit(iris.data[:,:2], iris.target)

plot_decision_boundary(knn_clf_all, axis=[4,8.5,1.5,4.5])
plt.scatter(iris.data[iris.target==0,0], iris.data[iris.target==0,1])
plt.scatter(iris.data[iris.target==1,0], iris.data[iris.target==1,1])
plt.scatter(iris.data[iris.target==2,0], iris.data[iris.target==2,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 调参
knn_clf_all = KNeighborsClassifier(n_neighbors=50)
knn_clf_all.fit(iris.data[:,:2], iris.target)


# ------------------------------- #
# 在逻辑回归中使用多项式特征
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X = np.random.normal(0,1,size=(200,2))
y = np.array(X[:,0]**2 + X[:,1]**2 < 1.5, dtype='int')

plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

# 使用逻辑回归
from LogisticRegression import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X,y)

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

plot_decision_boundary(log_reg, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

# 多项式项的逻辑回归
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X,y)

poly_log_reg.score(X, y)

plot_decision_boundary(poly_log_reg, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()


# ------------------------------- #
# sklearn中的逻辑回归
# ------------------------------- #
np.random.seed(666)
X = np.random.normal(0,1,size=(200,2))
y = np.array(X[:,0]**2 + X[:,1] < 1.5, dtype='int')
for _ in range(20):
    y[np.random.randint(200)] = 1

plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

from sklearn.model_selection import train_test_split
X_trian, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_trian, y_train)

log_reg.score(X_test, y_test)

plot_decision_boundary(log_reg, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

# 多项式项的逻辑回归
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_trian, y_train)

poly_log_reg.score(X_trian, y_train)
poly_log_reg.score(X_test, y_test)

plot_decision_boundary(poly_log_reg, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

# 正则化
def PolynomialLogisticRegression(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])

poly_log_reg3 = PolynomialLogisticRegression(degree=20, C=0.1)
poly_log_reg3.fit(X_trian, y_train)

poly_log_reg3.score(X_trian, y_train)
poly_log_reg3.score(X_test, y_test)

plot_decision_boundary(poly_log_reg3, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()

# penalty参数
def PolynomialLogisticRegression(degree, C, penalty='12'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C,penalty=penalty))
    ])

poly_log_reg4 = PolynomialLogisticRegression(degree=20, C=0.1, penalty='l1')
poly_log_reg4.fit(X_trian, y_train)


poly_log_reg4.score(X_trian, y_train)
poly_log_reg4.score(X_test, y_test)

plot_decision_boundary(poly_log_reg4, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1],color='r')
plt.scatter(X[y==1,0], X[y==1,1],color='b')
plt.show()


# ------------------------------- #
# OvR与OvO
# ------------------------------- #
# OvR(One vs Rest)一针对剩余: 默认OvR

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

from sklearn.model_selection import train_test_split
X_trian, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_trian, y_train)

log_reg.score(X_test, y_test)

plot_decision_boundary(log_reg, axis=[4,8.5,1.5,4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

# OvO(One vs One)一对一
log_reg2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg2.fit(X_trian, y_train)

log_reg2.score(X_test, y_test)

plot_decision_boundary(log_reg2, axis=[4,8.5,1.5,4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

# 使用所有数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_trian, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()
log_reg.fit(X_trian, y_train)

log_reg.score(X_test, y_test)

# OvO(One vs One)一对一
log_reg2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg2.fit(X_trian, y_train)
log_reg2.score(X_test, y_test)

# OvO and OvR
from sklearn.multiclass import OneVsRestClassifier

ovr = OneVsRestClassifier(log_reg)
ovr.fit(X_trian, y_train)
ovr.score(X_test, y_test)

from sklearn.multiclass import OneVsOneClassifier

ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_trian, y_train)
ovo.score(X_test, y_test)

