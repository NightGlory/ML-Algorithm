import numpy as np

# 查看版本号
print(np.__version__)

'''
nparr = np.array([i for i in range(10)])
print(nparr)

print(nparr.dtype)

nparr[3] = 1
print(nparr)

nparr2 = np.array([1,2,3.0])
print(nparr2.dtype)

nparr[3] = 3
print(nparr2)
print(nparr2.dtype)
'''

# numpy.zero()创建矩阵
print(np.zeros(10))
print(np.zeros(10, dtype=int))

print(np.zeros(shape=(3,5), dtype=int))

# numpy.ones()
print(np.ones((3,5),dtype=int))

# numpy.full()创建特定值的矩阵
print(np.full(shape=(3,5), fill_value=66))

# numpy.arange()
print(np.arange(0,20,2))

# numpy.linspace
print(np.linspace(0,20,11))

# numpy.random()
print(np.random.randint(0,10,size=10))

print(np.random.randint(0,10, size=(3,5)))

# numpy.random.seed()存储随机矩阵
np.random.seed(1)
print(np.random.randint(3,8,size=(3,5)))

np.random.seed(1)
print(np.random.randint(3,8,size=(3,5)))

# [0,1)之间的随机浮点数numpy.random.random()
print(np.random.random())

print(np.random.random(5))

print(np.random.random((3,5)))

# 符合标准正态分布的随机数numpy.random.normal()
print(np.random.normal())
# 设置为均值为10方差为100
print(np.random.normal(10,100))
# 可调整大小
print(np.random.normal(0,1,size=(3,5)))

# 查看numpy方法
dir(np.random.normal())
help(np.random.normal)

# 数组基本操作
x = np.arange(15).reshape(3,5)

# 几维数组
x.ndim

# 维度
x.shape

# 元素数
x.size

# numpy.array的数据访问
x[0]
x[-1]
x[0][0]
x[(0,0)]
x[2,2]
x[0:2]
x[::2]
x[::-1]
x[:2,:3]
x[:2][:3]   # 与上一个不一样

# 修改
subX = x[:2,:3]
subX[0,0] = 100
subX
x   # 同时被修改，因为numpy优先考虑效率

# 创建不相关的矩阵
subX = x[:2,:3].copy()
subX

# reshape
x.shape
x.reshape(1,15)
x.reshape(15,-1)
x.reshape(-1,15)

# 合并操作
x = np.array([1,2,3])
y = np.array([3,2,1])
np.concatenate([x,y])

z = np.array([666,666,666])
np.concatenate([x,y,z])

A = np.array([[1,2,3], [4,5,6]])
# 横向拼接，默认纵向拼接
np.concatenate([A,A], axis=1)

# 不同维度的矩阵拼接
np.concatenate([A,z.reshape(1,-1)])
np.vstack([A,z])    # 智能判断
B = np.full((2,2), 100)
np.hstack([A,B])    # 横向智能拼接

# 数据分割操作
x = np.arange(10)
x1,x2,x3 = np.split(x, [3,7])   # 分割点

A = np.arange(16).reshape((4,4))
A1,A2 = np.split(A,[2])

# 纵向分割
A1,A2 = np.split(A,[2], axis=1)
left, right = np.hsplit(A,[2]) 
# 横向分割
upper, lower = np.vsplit(A,[2])

# 常用分割数据（数据：标签）
X, y = np.hsplit(data,[-1])

# 矩阵运算(非numpy)
n = 10
L = (i for i in range(n))
A = []
for e in L:
    A.append(e*2)
# 或者
A = [2*e for e in L]

# 使用numpy.array实现矩阵运算
L = np.arange(n)
A = np.array(2*e for e in L)
# 或者
A = 2*L # numpy可用于所有的Universal Function

# 矩阵运算示例
A = np.arange(4).reshape(2,2)
B = np.full((2,2), 10)
A + B
A - B
A * B   # 非矩阵乘法
A.dot(B)    # 矩阵乘法·真
# 矩阵转置 X.T
A.T
C = np.full((3,3), 666)

# 向量矩阵的运算
v = np.array([1,2])
v + A
# 或者
np.vstack([v] * A.shape[0]) + A
# 或者
np.tile(v, (2,1)) + A

# 矩阵 * 向量
v.dot(A)
A.dot(v)    # 自动判断v为列向量

# 矩阵的逆
invA = np.linalg.inv(A)
A.dot(invA)

X = np.arange(16).reshape((2,8))
pinvX = np.linalg.pinv(X)   # 伪逆矩阵
pinvX.shape
X.dot(pinvX)

# 聚合操作
L = np.random.random(100)
np.sum(L)
np.min(L)
np.max(L)
X = np.arange(16).reshape(4,-1)
np.sum(X, axis=0)   # 沿列累加
np.prod(X + 1)  # 累乘
np.mean(X)  # 求平均值
np.median(X)    # 求中位数
np.percentile(X, q=50)  # L中50%的值，即中位数.q代表n分位数

# 方差、标准差
np.var(L)   #方差
np.std(L)   #标准差

# 例如
x = np.random.normal(0,1, size=100000)
np.mean(x)  # 均值
np.std(x)   # 标准差
np.var(x)   # 方差

# 索引(arg代表索引)
np.min(x)
np.argmin(x)
x[74136]
np.argmax(x)
np.max(x)

# 排序和使用索引
x = np.arange(16)
np.random.shuffle(x)    # 打乱
np.sort(x)  #排序

# 对于二维矩阵
X = np.random.randint(10, size=(4,4))
np.sort(X)  # 每一行排序
np.sort(X, axis=1)  # 默认为1
np.sort(X, axis=0)  # 每一列排序

# numpy.argsort()返回排序后的索引
np.argsort(x)
np.argsort(X)   # 每行索引排序
np.argsort(X, axis=0)

# 分类
np.partition(x, 3)  # 按n=3分出两个部分（大于n和小于n）
np.argpartition(x, 3)
np.argpartition(X, 2, axis=1)