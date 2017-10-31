import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

# 用 NumPy 随机生成 100 个数据
x_data = np.float32(np.random.rand(2, 100)) 
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
# tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
# tf.reduce_mean(x, 0) ==> [2.,  3.] #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
# tf.reduce_mean(x, 1) ==> [1.5,  3.5] #指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
# loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
session = tf.Session()
session.run(init)

# 拟合平面
for step in range(0, 401):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(W), session.run(b))
