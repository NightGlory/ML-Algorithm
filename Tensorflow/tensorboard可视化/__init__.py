import tensorboard
import tensorflow as tf

def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' %n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # 在 layer 中为 Weights, biases 设置变化图表
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        
        tf.summary.histogram(layer_name+'/outputs', outputs)

        return outputs

# 建造神经网络
import numpy as np

# make up some data
x_data= np.linspace(-1, 1, 300, dtype=np.float32)[:,np.newaxis]
noise=  np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data= np.square(x_data) -0.5+ noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 隐藏层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.reduce_sum(
        tf.square(ys - prediction),
        reduction_indices=[1]
    ))
    # 设置loss的变化图
    tf.summary.scalar('loss', loss)

# 优化器，学习率0.1
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# 给所有训练图合并
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs2/", sess.graph)
sess.run(tf.global_variables_initializer())

# 训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)