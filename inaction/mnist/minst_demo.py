import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets
import time

time_start=time.time()

(x, y), (x_test, y_test) = datasets.mnist.load_data()

x = tf.keras.utils.normalize(x)

train_data = tf.data.Dataset.from_tensor_slices((x, y))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def pretreatment(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


train_data = train_data.shuffle(1000).batch(128).map(pretreatment)

# 层数，节点数自定义，784 => 512 => 128 => 分类结果=10
# Variable的含义就是可更新的参数，所以必须严格定义为Variable
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([128]))
b3 = tf.Variable(tf.zeros([10]))

# 学习率 0.001
lr = 1e-3

# tape.watch([w1, b1, w2, b2, w3, b3]) Variable不需要
for epoch in range(20):
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape(persistent=True) as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3

            # loss: mse
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - out)))

            # 求梯度，求导
            with tf.device('/GPU:0'):
                grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

            # 更新参数，也就是前向传播参数
            # 按上面传入的参数顺序取值
            w1.assign_sub(grads[0] * lr)
            b1.assign_sub(grads[1] * lr)
            w2.assign_sub(grads[2] * lr)
            b2.assign_sub(grads[3] * lr)
            w3.assign_sub(grads[4] * lr)
            b3.assign_sub(grads[5] * lr)

            if step % 100 == 0:
                print(epoch, step, loss)

time_end = time.time()
print('cost time = ', time_end - time_start)