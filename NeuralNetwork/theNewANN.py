from tqdm import tqdm_notebook       # jupter notebook进度条
import matplotlib.pyplot as plt
import struct
import numpy as np
import math
from pathlib import Path
import copy
import pickle

# 定义激活函数


def bypass(x):
    return x


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()


def d_bypass(x):
    return 1

# softmax导数


def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm, sm)

# def d_tanh(data):
#     return np.diag(1/(np.cosh(data))**2)

# tanh导数


def d_tanh(data):
    return 1/(np.cosh(data))**2


d_type = {bypass: 'times', softmax: 'dot', tanh: 'times'}

differential = {softmax: d_softmax, tanh: d_tanh, bypass: d_bypass}

# 设参
dimensions = [28*28, 100, 10]
activation = [bypass, tanh, softmax]
distribution = [
    {},  # leave it empty!!
    {'b': [0, 0], 'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])),
                       math.sqrt(6/(dimensions[0]+dimensions[1]))]},
    {'b': [0, 0], 'w':[-math.sqrt(6/(dimensions[1]+dimensions[2])),
                       math.sqrt(6/(dimensions[1]+dimensions[2]))]},
]
batch_size = 100                # 批次


# 偏置b的初始化


def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]

# 权重w的初始化


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1], dimensions[layer])*(dist[1]-dist[0])+dist[0]


def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter


parameters = init_parameters()

# 预测


def predict(img, parameters):
    l_in = img
    l_out = activation[0](l_in)
    for layer in range(1, len(dimensions)):
        l_in = np.dot(l_out, parameters[layer]['w'])+parameters[layer]['b']
        l_out = activation[layer](l_in)
    return l_out


# 导入数据集
dataset_path = Path('./MNIST_data')
train_img_path = dataset_path/'train-images-idx3-ubyte'
train_lab_path = dataset_path/'train-labels-idx1-ubyte'
test_img_path = dataset_path/'t10k-images-idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels-idx1-ubyte'

train_num = 50000   # 训练集
valid_num = 10000   # 验证集
test_num = 10000    # 测试集

with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
    # 分给训练集和验证集
    train_img = tmp_img[:train_num]
    valid_img = tmp_img[train_num:]

with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255

with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    # 分给训练集和验证集
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]

with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)

# 绘制训练集的图


def show_train(index):
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(train_lab[index]))


def show_test(index):
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(test_lab[index]))


def show_valid(index):
    plt.imshow(valid_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(valid_lab[index]))


onehot = np.identity(dimensions[-1])    # 数值为1的10*10对角矩阵

# 平方误差


def sqrt_loss(img, lab, parameters):
    y_pred = predict(img, parameters)
    y = onehot[lab]
    diff = y-y_pred
    return np.dot(diff, diff)


def grad_parameters(img, lab, parameters):
    l_in_list = [img]
    l_out_list = [activation[0](l_in_list[0])]
    for layer in range(1, len(dimensions)):
        l_in = np.dot(l_out_list[layer-1],
                      parameters[layer]['w'])+parameters[layer]['b']
        l_out = activation[layer](l_in)
        l_in_list.append(l_in)
        l_out_list.append(l_out)

    d_layer = -2*(onehot[lab]-l_out_list[-1])

    grad_result = [None]*len(dimensions)
    for layer in range(len(dimensions)-1, 0, -1):
        if d_type[activation[layer]] == 'times':
            d_layer = differential[activation[layer]](l_in_list[layer])*d_layer
        if d_type[activation[layer]] == 'dot':
            d_layer = np.dot(differential[activation[layer]](
                l_in_list[layer]), d_layer)
        grad_result[layer] = {}
        grad_result[layer]['b'] = d_layer
        grad_result[layer]['w'] = np.outer(l_out_list[layer-1], d_layer)
        d_layer = np.dot(parameters[layer]['w'], d_layer)

    return grad_result


parameters = init_parameters()

""" # 验证
h = 0.001
layer = 1
pname = 'b'
grad_list = []
for i in range(len(parameters[layer][pname])):
    img_i = np.random.randint(train_num)
    test_parameters = init_parameters()
    derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)[layer][pname]
    value1 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
    test_parameters[layer][pname][i] += h
    value2 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
    grad_list.append(derivative[i]-(value2-value1)/h)
np.abs(grad_list).max()

h=0.00001
layer=2
pname='w'
grad_list=[]
for i in range(len(parameters[layer][pname])):
    for j in range(len(parameters[layer][pname][0])):
        img_i=np.random.randint(train_num)
        test_parameters=init_parameters()
        derivative=grad_parameters(train_img[img_i],train_lab[img_i],test_parameters)[layer][pname]
        value1=sqrt_loss(train_img[img_i],train_lab[img_i],test_parameters)
        test_parameters[layer][pname][i][j]+=h
        value2=sqrt_loss(train_img[img_i],train_lab[img_i],test_parameters)
        grad_list.append(derivative[i][j]-(value2-value1)/h)
np.abs(grad_list).max()
 """

# 验证函数


def valid_loss(parameters):         # loss_accumulate = (y1-y_pred_1)**2 + ... + (yn-y_pred_n)**2
    loss_accumulate = 0
    for img_i in range(valid_num):
        loss_accumulate += sqrt_loss(valid_img[img_i],
                                     valid_lab[img_i], parameters)
    return loss_accumulate/(valid_num/10000)


def valid_accuracy(parameters):
    correct = [predict(valid_img[img_i], parameters).argmax()
               == valid_lab[img_i] for img_i in range(valid_num)]
    # print('validation accuracy: {}'.format(correct.count(True) / len(correct)))
    return correct.count(True) / len(correct)


def train_loss(parameters):
    loss_accumulate = 0
    for img_i in range(train_num):
        loss_accumulate += sqrt_loss(train_img[img_i],
                                     train_lab[img_i], parameters)
    return loss_accumulate/(train_num/10000)


def train_accuracy(parameters):
    correct = [predict(train_img[img_i], parameters).argmax()
               == train_lab[img_i] for img_i in range(valid_num)]

    return correct.count(True) / len(correct)


def grad_add(grad1, grad2):
    for layer in range(1, len(grad1)):
        for pname in grad1[layer].keys():
            grad1[layer][pname] += grad2[layer][pname]
    return grad1


def grad_divide(grad, denominator):
    for layer in range(1, len(grad)):
        for pname in grad[layer].keys():
            grad[layer][pname] /= denominator
    return grad


def train_batch(current_batch, parameters):
    # 第0个，免去初始化
    grad_accumulate = grad_parameters(
        train_img[current_batch*batch_size+0], train_lab[current_batch*batch_size+0], parameters)

    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(
            train_img[current_batch*batch_size+img_i], train_lab[current_batch*batch_size+img_i], parameters)
        grad_add(grad_accumulate, grad_tmp)           # 累加
    grad_divide(grad_accumulate, batch_size)          # 求平均
    return grad_accumulate


def combine_parameters(parameters, grad, learn_rate):       # 调整b0, b1, w1
    parameter_tmp = copy.deepcopy(parameters)
    for layer in range(len(parameter_tmp)):
        for pname in parameter_tmp[layer].keys():
            parameter_tmp[layer][pname] -= learn_rate * grad[layer][pname]
    return parameter_tmp


parameters = init_parameters()  # 初始化
current_epoch = 0
train_loss_list = []
valid_loss_list = []
train_accuracy_list = []
valid_accuracy_list = []

# 训练前的准确率
# print('训练前的准确率:')
# valid_accuracy(parameters)

learn_rate = 1
epoch_num = 10       # 训练次数
for epoch in tqdm_notebook(range(epoch_num)):
    for i in range(train_num//batch_size):          # 训练了500次
        if i % 100 == 99:
            print('running batch {}/{}'.format(i+1, train_num//batch_size))
        grad_tmp = train_batch(i, parameters)
        parameters = combine_parameters(parameters, grad_tmp, learn_rate)

    current_epoch += 1
    train_loss_list.append(train_loss(parameters))
    train_accuracy_list.append(train_accuracy(parameters))
    valid_loss_list.append(valid_loss(parameters))
    valid_accuracy_list.append(valid_accuracy(parameters))

# 训练后的准确率
# print('训练后的准确率:')
# valid_accuracy(parameters)

# 绘制损失与精确度
lower = 0
plt.plot(valid_loss_list[lower:], color='red', label='validation loss')
plt.plot(train_loss_list[lower:], color='green', label='train loss')
plt.show()

plt.plot(valid_accuracy_list[lower:], color='red', label='valid accuracy')
plt.plot(train_accuracy_list[lower:], color='green', label='train accuracy')
plt.show()

# 学习率lr对损失函数和精确度的影响
rand_batch = np.random.randint(train_num//batch_size)
grad_lr = train_batch(rand_batch, parameters)
lr_list = []
lower = -2
upper = 0
step = 0.1
for lr_pow in np.linspace(lower, upper, num=(upper-lower)//step+1):
    learn_rate = 10**lr_pow
    parameter_tmp = combine_parameters(parameters, grad_lr, learn_rate)
    train_loss_tmp = train_loss(parameter_tmp)
    lr_list.append([lr_pow, train_loss_tmp])

# 学习率导致loss变化的绘图
upper = len(lr_list)
plt.plot(np.array(lr_list)[:upper, 0], np.array(
    lr_list)[:upper, 1], color='green')
plt.show()

# 导入模型
path = 'model/modelv2_01.pkl'
with open(path, 'rb') as f:
    (parameters,
    current_epoch,
    train_loss_list,
    valid_loss_list,
    train_accuracy_list,
    valid_accuracy_list
    ) = pickle.load(f)

valid_accuracy(parameters)
train_accuracy(parameters)

lower = 20
plt.plot(valid_accuracy_list[lower:], color='green', label='validation loss')
plt.plot(train_accuracy_list[lower:], color='red', label='train loss')
plt.show()

lower = 20
plt.plot(valid_loss_list[lower:], color='green', label='validation loss')
plt.plot(train_loss_list[lower:], color='red', label='train loss')
plt.show()

# 运行测试集
def test_accuracy(parameter):
    correct = [predict(test_img[img_i], parameters).argmax()
               == test_lab[img_i] for img_i in range(test_num)]

    return correct.count(True) / len(correct)

test_accuracy(parameters)