#---------------------#
# 批标准化 (Batch Normalization)
#---------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## 搭建网络
ACTIVATION = tf.nn.relu # 每一层都使用 relu 
N_LAYERS = 7            # 一共7层隐藏层
N_HIDDEN_UNITS = 30     # 每个层隐藏层有 30 个神经元

def fix_seed(seed=1):
# reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)

def plot_his(inputs, inputs_norm):
# plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
            plt.draw()
            plt.pause(0.01)

def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        # 添加层功能
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        #BN
        if norm:    # 判断输入否是 BN 层
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],   # 想要 normalize 的维度, [0] 代表 batch 维度
                # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    fix_seed(1)

    if norm:    # 判断输入否是 BN 层
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],   # 想要 normalize 的维度, [0] 代表 batch 维度
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        Wx_plus_b = tf.nn.batch_normalization(xs, fc_mean, fc_var, shift, scale, epsilon)
        # 上面那一步, 在做如下事情:
        # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
        # Wx_plus_b = Wx_plus_b * scale + shift

    layers_inputs = [xs]    # 记录每层的 input

    # loop 建立所有层
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
            norm,
        )
        layers_inputs.append(output)    # 把 output 加入记录

    # 建立 output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]

## 创建数据
x_data = np.linspace(-7, 10, 500)[:, np.newaxis]
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise

# 可视化 input data
plt.scatter(x_data, y_data)
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features]
ys = tf.placeholder(tf.float32, [None, 1])  


train_op, cost, layers_inputs = built_net(xs, ys, norm=False)   # without BN
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True) # with BN

# record cost 训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 记录两种网络的 cost 变化
cost_his = []
cost_his_norm = []
record_step = 5

plt.ion()
plt.figure(figsize=(7, 3))
for i in range(251):
    if i % 50 == 0:
        # 每层在 activation 之前计算结果值的分布
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        plot_his(all_inputs, all_inputs_norm)

    sess.run(train_op, feed_dict={xs: x_data, ys: y_data})
    sess.run(train_op_norm, feed_dict={xs: x_data, ys: y_data})
    if i % record_step == 0:
        # 记录 cost
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
plt.legend()
plt.show()