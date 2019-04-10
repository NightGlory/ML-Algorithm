#---------------------#
# Classification 分类学习
#---------------------#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs

# 精确度计算
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10]) # 10个输出

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))   # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess =tf.Session()
sess.run(tf.global_variables_initializer())

# 训练
for i in range(1000):
    # 现在开始train，每次只取100张图片，免得数据太多训练太慢。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))


#---------------------#
# Overfitting 过拟合
#---------------------#
# Dropout解决Overfitting
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

tf.reset_default_graph()

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

def add_layer(inputs, in_size, out_size,n_layer, activation_function = None):
    layer_name = 'layer%s' %n_layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) # 8*8
ys = tf.placeholder(tf.float32, [None, 10]) # 10个输出

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))   # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess =tf.Session()
merged = tf.summary.merge_all()

# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
