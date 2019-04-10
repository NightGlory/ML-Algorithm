# ------------------------------- #
# PCA主成分分析
# ------------------------------- #

# 作用：降维
# 步骤：
# 1. 对所有的样本进行demean处理
# 2. 求一个轴的方向 w=(w1, w2)，使得所有的样本，映射到w以后，有var(X) = 1/m * sum^m_i=1 ||X_i - X_avg||^2
# 目标函数：var(X) = 1/m * sum (X_i.dot(w))^2,使用梯度上升法解决
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75*X[:,0]+3.+np.random.normal(0,10.,size=100)

plt.scatter(X[:,0], X[:,1])
plt.show()

# demean
def demean(X):
    return X - np.mean(X, axis=0)   #行方向求均值

X_demean = demean(X)

plt.scatter(X_demean[:,0], X_demean[:,1])
plt.show()

# 验证
np.mean(X_demean[:,0])
np.mean(X_demean[:,1])

# 梯度上升法
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df_math(w,X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

# 验证
def df_debug(w, X, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res

def direction(w):
    return w / np.linalg.norm(w)    #np.linalg.norm(w): 求模

# 梯度上升
def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)        # 注意1: 每次求一个单位向量
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break        
        cur_iter += 1
    return w

initial_w = np.random.random(X.shape[1])    # 注意2: 不能用0向量开始
eta = 0.001

# 注意3: 不能使用StandardScaler标准化数据
# 检验
gradient_ascent(df_debug, X_demean, initial_w, eta)

# 实际
gradient_ascent(df_math, X_demean, initial_w, eta)

# 可视化
w = gradient_ascent(df_math, X_demean, initial_w, eta)
plt.scatter(X_demean[:,0], X_demean[:,1])
plt.plot([0, w[0]*30], [0, w[1]*30], color='r')
plt.show()


# ------------------------------- #
# 求数据前的n个主成分
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75*X[:,0]+3.+np.random.normal(0,10.,size=100)

plt.scatter(X[:,0], X[:,1])
plt.show()

# demean
def demean(X):
    return X - np.mean(X, axis=0)   #行方向求均值

X_demean = demean(X)

# 梯度上升法
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w,X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)    #np.linalg.norm(w): 求模

# 梯度上升
def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)        # 注意1: 每次求一个单位向量
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break        
        cur_iter += 1
    return w

initial_w = np.random.random(X.shape[1])
eta = 0.01
w = first_component(X, initial_w, eta)

X2 = np.empty(X.shape)
for i in range(len(X)):
    X2[i] = X[i] - X[i].dot(w) * w
plt.scatter(X2[:,0], X2[:,1])
plt.show()

# 或者
X2 = X - X.dot(w).reshape(-1,1) * w

w2 = first_component(X2, initial_w, eta)
w2

w.dot(w2)

# 封装成函数
def first_n_component(n, X, eta=0.01, n_iters=1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)

        X_pca = X_pca - X_pca.dot(w).reshape(-1,1) * w
    
    return res

first_n_component(2, X)


# ------------------------------- #
# 高维数据映射为低维数据
# ------------------------------- #

# 测试封装后的类
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75*X[:,0]+3.+np.random.normal(0,10.,size=100)

from PCA import PCA

pca = PCA(n_components=2)
pca.fit(X)

pca.components_

pca = PCA(n_components=1)
pca.fit(X)

X_reduction = pca.transform(X)
X_reduction.shape

X_restore = pca.inverse_transform(X_reduction)
X_restore.shape

plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()


# ------------------------------- #
# scikit-learn中的PCA
# ------------------------------- #
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(X)

pca.components_

X_reduction = pca.transform(X)
X_reduction.shape

X_restore = pca.inverse_transform(X_reduction)
X_restore.shape

plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()

# 真实数据集
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

X_train.shape

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

knn_clf.score(X_test, y_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)

knn_clf.score(X_test_reduction, y_test)

# 总结：维度从64维度降到2维度，运行速度提升但预测精度降低

# 网格搜索查询最优降低维度
# 但是sklearn提供了查询降维精度方法
pca.explained_variance_ratio_

# 多组查询
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
pca.explained_variance_ratio_

plt.plot([i for i in range(X_train.shape[1])], 
        [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()

# PCA(n) n为精度
pca = PCA(0.95)
pca.fit(X_train)
pca.n_components_   # 维度

# 新的训练
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
knn_clf.score(X_test_reduction, y_test)

# 降维到2维的意义是可视化
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)

for i in range(10):
    plt.scatter(X_reduction[y==i,0], X_reduction[y==i,1], alpha=0.8)
plt.show()


# ------------------------------- #
# MNIST
# ------------------------------- #
import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")

X, y = mnist['data'], mnist['target']

X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

# PCA 降维
from sklearn.decomposition import PCA

pca = PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_train_reduction.shape


# ------------------------------- #
# 使用PCA对数据进行降噪
# ------------------------------- #
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target

noisy_digits = X + np.random.normal(0, 4, size=X.shape)

example_digits = noisy_digits[y==0,:][:10]
for num in range(1, 10):
    X_num = noisy_digits[y==num,:][:10]
    example_digits = np.vstack([example_digits, X_num])

example_digits.shape

# 绘制图像
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10,10),
                subplot_kw={'xticks':[], 'yticks':[]},
                gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),
                cmap='binary', interpolation='nearest',
                clim=(0, 16))
    plt.show()

plot_digits(example_digits)

# 降噪
from sklearn.decomposition import PCA

pca = PCA(0.5)
pca.fit(noisy_digits)
pca.n_components_

components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)


# ------------------------------- #
# 人脸识别合特征脸
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people()

faces.keys()
faces.data.shape
faces.images.shape

random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]

example_faces = X[:36, :]
example_faces.shape

# 绘制
def plot_faces(faces):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                subplot_kw={'xticks':[], 'yticks':[]},
                gradspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()

plot_faces(example_faces)

faces.target_names

len(faces.target_names)

# 特征脸
from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized')
pca.fit(X)

pca.components_.shape

plot_faces(pca.components_[:36, :])

# fetch_lfw_people库
faces2 = fetch_lfw_people(min_faces_per_person=60)   # 一个人至少60张图片的数据库