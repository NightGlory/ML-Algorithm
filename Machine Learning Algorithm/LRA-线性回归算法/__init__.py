# ------------------------------- #
# 线性回归算法
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# 可视化
plt.scatter(x, y)
plt.axis([0,6,0,6])
plt.show()

# 最小二乘法得到a和b
x_mean = np.mean(x)
y_mean = np.mean(y)
num = 0.0   # 分子
d = 0.0     # 分母
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
a = num / d
b = y_mean - a * x_mean

# 可视化
y_hat = a * x + b
plt.scatter(x,y)
plt.plot(x, y_hat, color='r')
plt.axis([0,6,0,6])
plt.show()

x_predict = 6
y_predict = a * x_predict + b

# 使用封装好的类
from SimpleLinearRegression import SimpleLinearRegression1

reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
reg1.predict(np.array([x_predict]))
reg1.a_,reg1.b_ # 查看参数

# 可视化
y_hat1 = reg1.predict(x)
plt.scatter(x,y)
plt.plot(x, y_hat1, color='g')
plt.axis([0,6,0,6])
plt.show()


# ------------------------------- #
# 向量化运算
# ------------------------------- #
from SimpleLinearRegression import SimpleLinearRegression2
import timeit

reg2 = SimpleLinearRegression2()
reg2.fit(x, y)

reg2.a_

# 向量化实现的性能测试
m = 1000000
big_x = np.random.random(size=m)
big_y = big_x * 2.0 + 3.0 + np.random.normal(size=m)

# SimpleLinearRegression1
begin1 = timeit.default_timer() # 起始时间
reg1.fit(big_x, big_y)
end1 = timeit.default_timer()   # 终结时间
runtime1 = end1 - begin1
print("SimpleLinearRegression1: ", round(runtime1, 4), "s")

# SimpleLinearRegression2
begin2 = timeit.default_timer() # 起始时间
reg2.fit(big_x, big_y)
end2 = timeit.default_timer()   # 终结时间
runtime2 = end2 - begin2
print("SimpleLinearRegression2: ", round(runtime2, 4), "s")

# 判断
if runtime1 >= runtime2:
    print("SimpleLinearRegression2优于SimpleLinearRegression1")
else:
    print("SimpleLinearRegression1优于SimpleLinearRegression2")


# ------------------------------- #
# 衡量线性回归算法的指标：MSE,RMSE,MAE
# ------------------------------- #

# MSE: Mean Squared Error 均方误差
# RMSE: Root Mean Squared Error 均方根误差
# MAE: Mean Absolute Error 平均绝对误差

# 衡量回归算法的标准：以波士顿房产数据为例
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
print(boston.DESCR) # 查看相关数据集信息
boston.feature_names
x = boston.data[:, 5]   # 只使用房间数量这个特征
x.shape
y = boston.target
y.shape

# 可视化
plt.scatter(x, y)
plt.show()

x = x[y < np.max(y)]    #删除不可靠的点
y = y[y < np.max(y)]

# 简单线性回归算法
from sklearn.model_selection import train_test_split
from SimpleLinearRegression import SimpleLinearRegression2

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 666)
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)

# 可视化
plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color="r")
plt.show()

y_predict = reg.predict(x_test)

# MSE的计算
mse_test = np.sum((y_predict - y_test)**2) / len(y_test)

#RMSE的计算
from math import sqrt
rmse_test = sqrt(mse_test)

# MAE的计算
mae_test = np.sum(np.absolute(y_predict - y_test)) / len(y_test)

# 调用metrics.py
from metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

mean_squared_error(y_test, y_predict)

# scikit-learn中的MSE和MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error

mean_absolute_error(y_test, y_predict)


# ------------------------------- #
# 最好的衡量线性回归算法的指标：R Squared
# ------------------------------- #

# R^2 = 1 - MSE(y_hat,y) / Var(y)

R_Squared = 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)

from metrics import r2_score
r2_score(y_test, y_predict)

# scikit-learn中的r2_score
from sklearn.metrics import r2_score
r2_score(y_test, y_predict)

reg.score(x_test, y_test)


# ------------------------------- #
# 多元线性回归
# ------------------------------- #

from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y<50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 666)

from LinearRegression import LinearRegression

reg = LinearRegression()
reg.fit_normal(X_train, y_train)

# 评价
reg.score(X_test, y_test)


# ------------------------------- #
# scikit-learn中的回归问题
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y<50.0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 666)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.coef_
lin_reg.intercept_

lin_reg.score(X_test, y_test)

# kNN Regressor
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)
knn_reg.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV

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

knn_reg = KNeighborsRegressor()
gird_search = GridSearchCV(knn_reg, para_grid, n_jobs=-1, verbose=1)
gird_search.fit(X_train, y_train)

gird_search.best_params_
gird_search.best_score_
gird_search.best_estimator_.score(X_test, y_test)