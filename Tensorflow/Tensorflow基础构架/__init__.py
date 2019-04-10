#---------------------#
# 处理结构
#---------------------#
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  #反向传递误差
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# create tensorflow structure end
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(Weights), sess.run(biases))
    
sess.close()


#---------------------#
# Session 会话控制
#---------------------#
import tensorflow as tf

# create two matrixes
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2], 
                        [2]])
product = tf.matmul(matrix1, matrix2)   # matrix multiply = np.dot(m1, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


#---------------------#
# Variable 变量
#---------------------#
import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)
# 将 State 更新成 new_value
update = tf.assign(state, new_value)

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer()

# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


#---------------------#
# Placeholder 传入值
#---------------------#
import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))

