# ------------------------------- #
# 多项式回归
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = 0.5 * x**2 + x + 2 +np.random.normal(0,1, size=100)

# 可视化
plt.scatter(x, y)
plt.show()

# 若使用线性拟合,效果不好
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

y_predict = lin_reg.predict(X)

plt.scatter(x,y)
plt.plot(x, y_predict,color='r')
plt.show()

# 解决方案：添加一个特征
(X**2).shape

X2 = np.hstack([X, X**2])
X2.shape

lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)

# 可视化
plt.scatter(x,y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)],color='r')
plt.show()

# 查看系数
lin_reg2.coef_
lin_reg2.intercept_ # 截距


# ------------------------------- #
# scikit-learn中的多项式回归于pipeline
# ------------------------------- #
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)     # 参数：几次幂
poly.fit(X)
X2 = poly.transform(X)

X2.shape
X2[:5,:]

from sklearn.linear_model import LinearRegression

lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)

# 可视化
plt.scatter(x,y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)],color='r')
plt.show()

# 查看系数
lin_reg2.coef_
lin_reg2.intercept_ # 截距

# 关于PolynomialFeatures
X = np.arange(1,11).reshape(-1,2)

X.shape

poly = PolynomialFeatures(degree=3)
poly.fit(X)
X3 = poly.transform(X)

X3.shape

# Pipeline管道
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])

poly_reg.fit(X, y)


# ------------------------------- #
# 过拟合和欠拟合
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = 0.5 * x**2 + x + 2 +np.random.normal(0,1, size=100)

# 可视化
plt.scatter(x, y)
plt.show()

# 使用多项式回归
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

poly_reg = PolynomialRegression(degree=2)
poly_reg.fit(X, y)

y_predict = poly_reg.predict(X)
mean_squared_error(y, y_predict)

# 可视化
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.show()

# degree=10
poly10_reg = PolynomialRegression(degree=10)
poly10_reg.fit(X, y)

y10_predict = poly10_reg.predict(X)
mean_squared_error(y, y10_predict)

# 可视化
plt.scatter(x, y)
plt.plot(np.sort(x), y10_predict[np.argsort(x)], color='r')
plt.show()

#degree=100
poly100_reg = PolynomialRegression(degree=100)
poly100_reg.fit(X, y)

y100_predict = poly100_reg.predict(X)
mean_squared_error(y, y100_predict)

# 可视化
plt.scatter(x, y)
plt.plot(np.sort(x), y100_predict[np.argsort(x)], color='r')
plt.show()

# 实际曲线:过拟合
X_plot = np.linspace(-3, 3, 100).reshape(100,1)
y_plot = poly100_reg.predict(X_plot)

plt.scatter(x, y)
plt.plot(X_plot[:,0], y_plot, color='r')
plt.axis([-3, 3, -1, 10])
plt.show()


# ------------------------------- #
# train_test_split的意义
# ------------------------------- #
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 线性回归
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_predict = lin_reg.predict(X_test)
mean_squared_error(y_test, y_predict)

# 多项式回归
poly2_reg = PolynomialRegression(degree=2)
poly2_reg.fit(X_train, y_train)
y2_predict = poly2_reg.predict(X_test)
mean_squared_error(y_test, y2_predict)

# degree=10
poly10_reg = PolynomialRegression(degree=10)
poly10_reg.fit(X_train, y_train)
y10_predict = poly10_reg.predict(X_test)
mean_squared_error(y_test, y10_predict)

# degree=100
poly100_reg = PolynomialRegression(degree=100)
poly100_reg.fit(X_train, y_train)
y100_predict = poly100_reg.predict(X_test)
mean_squared_error(y_test, y100_predict)


# ------------------------------- #
# 学习曲线
# ------------------------------- #
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

train_score = []
test_score = []

for i in range(1, 76):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train[:i], y_train[:i])
    y_train_predict = lin_reg.predict(X_train[:i])
    train_score.append(mean_squared_error(y_train[:i], y_train_predict))
    y_test_predict = lin_reg.predict(X_test)
    test_score.append(mean_squared_error(y_test, y_test_predict))

plt.plot([i for i in range(1, 76)], np.sqrt(train_score), label='train')
plt.plot([i for i in range(1, 76)], np.sqrt(test_score), label='test')
plt.legend()
plt.show()

# 封装成函数
def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))
    
    plt.plot([i for i in range(1, len(X_train)+1)],
                            np.sqrt(train_score), label='train')
    plt.plot([i for i in range(1, len(X_train)+1)],
                            np.sqrt(test_score), label='test')
    plt.legend()
    plt.axis([0, len(X_train)+1, 0, 4])
    plt.show()

# 函数调用
plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)

# 二阶
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

poly2_reg = PolynomialRegression(degree=2)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)


# ------------------------------- #
# 验证数据集与交叉验证
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

from sklearn.neighbors import KNeighborsClassifier

best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score, best_p, best_k = score, p, k

# 使用交叉验证
from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier()
cross_val_score(knn_clf, X_train, y_train)

# 调参
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf ,X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k

print("Best K = ", best_k)
print("Best P = ", best_p)
print("Best Score = ", best_score)

# 验证
best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=3, p=2)
best_knn_clf.fit(X_train, y_train)
best_knn_clf.score(X_test, y_test)

# 回顾网格搜索
from sklearn.model_selection import GridSearchCV

para_grid = [
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(2,11)],
        'p': [i for i in range(1,6)]
    }
]

grid_search = GridSearchCV(knn_clf, para_grid, verbose=1)
grid_search.fit(X_train, y_train)

grid_search.best_score_
grid_search.best_params_

# 数据集分为5份
cross_val_score(knn_clf, X_train, y_train, cv=5)    


# ------------------------------- #
# 模型泛化与岭回归Ridge Regression
# ------------------------------- #
# 模型正则化Regularization
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1,1)
y = 0.5 * x + 3 + np.random.normal(0,1,size=100)

plt.scatter(x, y)
plt.show()

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

from sklearn.model_selection import train_test_split

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.metrics import mean_squared_error

poly10_reg = PolynomialRegression(degree=20)
poly10_reg.fit(X_train, y_train)

y_poly_predict = poly10_reg.predict(X_test)
mean_squared_error(y_test, y_poly_predict)

X_plot = np.linspace(-3,3,100).reshape(100,1)
y_plot = poly10_reg.predict(X_plot)

plt.scatter(x, y)
plt.plot(X_plot[:,0], y_plot, color='r')
plt.axis([-3,3,0,6])
plt.show()

def plot_model(model):
    X_plot = np.linspace(-3,3,100).reshape(100,1)
    y_plot = model.predict(X_plot)

    plt.scatter(x, y)
    plt.plot(X_plot[:,0], y_plot, color='r')
    plt.axis([-3,3,0,6])
    plt.show()

plot_model(poly10_reg)

# 使用岭回归
from sklearn.linear_model import Ridge

# ridge = Ridge(alpha=1)

def RidgeRegression(degree,alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])

ridge1_reg = RidgeRegression(20, 0.0001)
ridge1_reg.fit(X_train, y_train)

y1_predict = ridge1_reg.predict(X_test)
mean_squared_error(y_test, y1_predict)

plot_model(ridge1_reg)

ridge2_reg = RidgeRegression(20, 1)
ridge2_reg.fit(X_train, y_train)

y2_predict = ridge1_reg.predict(X_test)
mean_squared_error(y_test, y2_predict)

plot_model(ridge2_reg)  # 更加平滑


# ------------------------------- #
# LASSO 回归 Least Absolute and Selection Operator Regression
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1,1)
y = 0.5 * x + 3 + np.random.normal(0,1,size=100)

plt.scatter(x, y)
plt.show()

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

from sklearn.model_selection import train_test_split

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.metrics import mean_squared_error

poly_reg = PolynomialRegression(degree=20)
poly_reg.fit(X_train, y_train)

y_predict = poly_reg.predict(X_test)
mean_squared_error(y_test, y_predict)

def plot_model(model):
    X_plot = np.linspace(-3,3,100).reshape(100,1)
    y_plot = model.predict(X_plot)

    plt.scatter(x, y)
    plt.plot(X_plot[:,0], y_plot, color='r')
    plt.axis([-3,3,0,6])
    plt.show()

plot_model(poly_reg)

# LASSO
from sklearn.linear_model import Lasso

def LassoRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha))
    ])

lasso1_reg = LassoRegression(20, 0.01)
lasso1_reg.fit(X_train, y_train)

y1_predict = lasso1_reg.predict(X_test)
mean_squared_error(y_test, y1_predict)

# 可视化
plot_model(lasso1_reg)

# alpha=0.1
lasso2_reg = LassoRegression(20, 0.1)
lasso2_reg.fit(X_train, y_train)

y2_predict = lasso2_reg.predict(X_test)
mean_squared_error(y_test, y2_predict)

# 可视化
plot_model(lasso2_reg)

# 总结：LASSO趋向于使得一部分theta值变为0，所以可作为特征选择用


# ------------------------------- #
# L1， L2，弹性网络
# ------------------------------- #
# 弹性网:结合两种回归方法
