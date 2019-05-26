from tqdm import tqdm_notebook       # jupter notebook进度条
import matplotlib.pyplot as plt
import struct
import numpy as np
import math
from pathlib import Path
import copy

# 定义激活函数


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()


# 设参
dimensions = [784, 10]
activation = [tanh, softmax]
distribution = [
    {'b': [0, 0]},
    {'b': [0, 0], 'w': [-math.sqrt(6/(dimensions[0]+dimensions[1])),
                        math.sqrt(6/(dimensions[0]+dimensions[1]))]},
]

# 偏置b的初始化


def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer]) * (dist[1] - dist[0]) + dist[0]

# 权重w的初始化


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]


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
    # l0 = A1(x + b0)
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['w']) + \
        parameters[1]['b']     # l1 = A2(w*l0 + b1)
    l1_out = activation[1](l1_in)
    return l1_out


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


# softmax导数


def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm, sm)

# def d_tanh(data):
#     return np.diag(1/(np.cosh(data))**2)


def d_tanh(data):
    return 1/(np.cosh(data))**2


differential = {softmax: d_softmax, tanh: d_tanh}

""" 
验证导数
h=0.0001
func = tanh
input_len = 4
for i in range(input_len):
    test_input = np.random.rand(input_len)
    derivative = differential[func](test_input)
    value1 = func(test_input)
    test_input[i] += h
    value2 = func(test_input)
    # 误差
    print(derivative[i] - (value2-value1)/h) 
"""

onehot = np.identity(dimensions[-1])    # 数值为1的10*10对角矩阵

# 平方误差


def sqrt_loss(img, lab, parameters):
    y_pred = predict(img, parameters)
    y = onehot[lab]
    diff = y-y_pred
    return np.dot(diff, diff)


def grad_parameters(img, lab, parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['w']) + parameters[1]['b']
    l1_out = activation[1](l1_in)

    diff = onehot[lab] - l1_out
    act1 = np.dot(differential[activation[1]](l1_in), diff)

    grad_b1 = -2 * act1
    grad_w1 = -2 * np.outer(l0_out, act1)
    grad_b0 = -2 * differential[activation[0]
                                ](l0_in) * np.dot(parameters[1]['w'], act1)

    return {'w1': grad_w1, 'b1': grad_b1, 'b0': grad_b0}


""" 
# b1验证
h=0.00001
for i in range(10):
    img_i = np.random.randint(train_num)
    test_parameters = init_parameters()
    derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)['b1']
    value1 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
    test_parameters[1]['b'][i] += h
    value2 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
    print(derivative[i]-(value2 - value1)/h)
"""

""" 
# w1验证
grad_list = []
h=0.00001
for i in range(784):
    for j in range(10):
        img_i = np.random.randint(train_num)
        test_parameters = init_parameters()
        derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)['w1']
        value1 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
        test_parameters[1]['w'][i][j] += h
        value2 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
        grad_list.append(derivative[i][j]-(value2 - value1)/h)
np.abs(grad_list).max()
"""

""" 
# b0验证
grad_list = []
h=0.00001
for i in range(784):
    img_i = np.random.randint(train_num)
    test_parameters = init_parameters()
    derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)['b0']
    value1 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
    test_parameters[0]['b'][i] += h
    value2 = sqrt_loss(train_img[img_i], train_lab[img_i], test_parameters)
    grad_list.append(derivative[i]-(value2 - value1)/h)
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


batch_size = 100                # 批次


def train_batch(current_batch, parameters):
    # 第0个，免去初始化
    grad_accumulate = grad_parameters(
        train_img[current_batch*batch_size+0], train_lab[current_batch*batch_size+0], parameters)

    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(
            train_img[current_batch*batch_size+img_i], train_lab[current_batch*batch_size+img_i], parameters)
        for key in grad_accumulate.keys():
            grad_accumulate[key] += grad_tmp[key]   # 累加
    for key in grad_accumulate.keys():
        grad_accumulate[key] /= batch_size          # 求平均
    return grad_accumulate


def combine_parameters(parameters, grad, learn_rate):       # 调整b0, b1, w1
    parameter_tmp = copy.deepcopy(parameters)
    parameter_tmp[0]['b'] -= learn_rate * grad['b0']
    parameter_tmp[1]['b'] -= learn_rate * grad['b1']
    parameter_tmp[1]['w'] -= learn_rate * grad['w1']
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
