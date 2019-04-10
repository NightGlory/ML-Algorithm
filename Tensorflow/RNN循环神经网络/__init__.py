#---------------------#
# RNN循环神经网络(Recurrent Neural Network)
#---------------------#
# LSTM长短期记忆 循环神经网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

tf.set_random_seed(1)   # set random seed
tf.reset_default_graph()

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128            
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder / tf graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), 
    # shape(128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape(128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    # shape(10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

# 定义 RNN 的主体结构
def RNN(X, weights, biases):
    ## hidden layer for input to cell
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X(128 batches, 28 steps, 28 inputs) ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    ## cell
    # 使用 basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, forget_bias=1.0)
    # lstm cell is divided into two parts(c_state, m_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    
    ## hidden layer for output as the final results
    # 法一：
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    
    # 法二：把 outputs 变成 列表 [(batch, outputs)..] * steps
    #outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))    # states is the last outputs
    #results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output
    
    return results

# 计算 cost 和 train_op
with tf.variable_scope('pred'):
    pred = RNN(x, weights, biases)
with tf.variable_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.variable_scope('train_op'):
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

#训练 RNN
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x:batch_xs,
            y:batch_ys,
        })
        if step%20 == 0:
            print(sess.run(accuracy, feed_dict={
                x:batch_xs,
                y:batch_ys,
            }))
        step += 1
