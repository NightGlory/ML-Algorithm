# ------------------------------- #
# 梯度下降法
# ------------------------------- #

# 基于搜索的最优化算法。作用是最小化一个损失函数
# 对比梯度上升法：最大化效用函数

import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1,6,141)
plot_x
plot_y = (plot_x-2.5)**2-1

# 可视化
plt.plot(plot_x,plot_y)
plt.show()

# 导数定义
def dJ(theta):
    return 2*(theta-2.5)

# 损失函数
def J(theta):
    try:
        return (theta-2.5)**2-1
    except:
        return float('inf')

# 梯度下降法
eta = 0.1   # 学习率
epsilon = 1e-8  # 精度

theta = 0.0 # 超参数
theta_history = [theta] # 存储参数历史记录

while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient  # 更新
    theta_history.append(theta)

    if(abs(J(theta) - J(last_theta)) < epsilon):
        break

# 可视化theta_history
plt.plot(plot_x, J(plot_x))
# plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker="+")
plt.scatter(np.array(theta_history), J(np.array(theta_history)), color="r", marker="+")
plt.show()

len(theta_history)  # 次数

# 封装成函数
def gradient_descent(initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0

    while i_iter < n_iters:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient  # 更新
        theta_history.append(theta)

        if(abs(J(theta) - J(last_theta)) < epsilon):
            break
        
        i_iter += 1
    
    return

def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker="+")
    # plt.scatter(np.array(theta_history), J(np.array(theta_history)), color="r", marker="+")
    plt.show()

# 测试学习率eta
eta = 1.1
theta_history = []
gradient_descent(0., eta, n_iters=10)
plot_theta_history()


# ------------------------------- #
# 线性回归中的梯度下降法
# ------------------------------- #

# 目标：使得J(theta) = MSE(y, y_hat)尽可能小
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3 + 4. + np.random.normal(size=100)

X = x.reshape(-1, 1)

plt.scatter(x, y)
plt.show()

# 使用梯度下降法训练
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')

def dJ(theta, X_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
    return res * 2 / len(X_b)

def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    theta = initial_theta
    i_iter = 0

    while i_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient  # 更新

        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
        
        i_iter += 1
    
    return theta

X_b = np.hstack([np.ones((len(x), 1)), x.reshape(-1,1)])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01

theta = gradient_descent(X_b, y, initial_theta, eta)    # 对应截距和斜率

# 测试我们封装好的线性回归算法
from LinearRegression import LinearRegression

lin_gre = LinearRegression()
lin_gre.fit_gd(X, y)

lin_gre.coef_,lin_gre.intercept_


# ------------------------------- #
# 梯度下降的向量化和数据标准化
# ------------------------------- #

# tri_J(theta) = 2 * X_b.T * (X_b * theta - y) / m

from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y<50.0]
y = y[y<50.0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from LinearRegression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit_gd(X_train, y_train, eta=0.000001)
lin_reg.score(X_test, y_test)

# 使用梯度下降法前进行数据归一化
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)

lin_reg3 = LinearRegression()
# lin_reg3.fit_gd(X_train, y_train)
X_test_standard = standardScaler.transform(X_test)
lin_reg3.fit_gd(X_train_standard, y_train)
lin_reg3.predict(X_test_standard)
lin_reg3.score(X_test_standard, y_test)

# 梯度下降法优势
m=1000
n=5000
big_X = np.random.normal(size=(m,n))
true_theta = np.random.uniform(0.1, 100.0, size=n+1)
big_y = big_X.dot(true_theta[1:]) + true_theta[0] + np.random.normal(0.,10., size=m)

big_reg1 = LinearRegression()
big_reg1.fit_gd(big_X, big_y)

# ------------------------------- #
# 随机梯度下降法
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

m=100000

x = np.random.normal(size=m)
X = x.reshape(-1,1)
y = 4. * x + 3. +np.random.normal(0, 3, size=m)

def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    except:
        return float('inf')

def dJ_sgd(theta, X_b_i, y_i):  #改变
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

def sgd(X_b, y, initial_theta, n_iters):
    t0 = 5
    t1 = 50

    def learning_rate(t):
        return t0 / (t + t1)
    
    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(X_b))
        gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
        theta = theta - learning_rate(cur_iter) * gradient
    
    return theta

import timeit

begin = timeit.default_timer()

X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
theta = sgd(X_b, y, initial_theta, n_iters=len(X_b)//3)

end = timeit.default_timer()

print("Run time: ", round(end-begin, 6),"s")

# 测试封装的函数
from LinearRegression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit_sgd(X, y, n_iters=2)
lin_reg.score(X, y)

# 使用真实数据测试
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y<50.]
y = y[y<50.]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 归一化
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

from LinearRegression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit_sgd(X_train_standard, y_train, n_iters=100)
lin_reg.score(X_test_standard, y_test)

# scikit-learn中的SGD
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=100)
sgd_reg.fit(X_train_standard, y_train)
sgd_reg.score(X_test_standard, y_test)

# ------------------------------- #
# 如何确定梯度计算的准确性 调试梯度下降法
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X = np.random.random(size=(1000, 10))
true_theta = np.arange(1,12, dtype=float)

X_b = np.hstack([np.ones((len(X), 1)),X])
y = X_b.dot(true_theta) + np.random.normal(size=1000)

def J(theta, X_b, y):
    try:
        return np.sum((y-X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')
    
def dJ_math(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta)-y)*2. /len(y)

def dJ_debug(theta, X_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2*epsilon)
    
    return res

