import tensorflow as tf
import numpy as np

a = tf.cast(tf.reshape(tf.range(100), [2, 5, 10]), dtype=tf.float32)
print(a)

print(tf.reduce_max(a, axis=0))
print(tf.reduce_min(a, axis=0))
print('均值：', tf.reduce_mean(a, axis=0))
print(tf.reduce_sum(a, axis=0))

print('---------axis=1--------')

print(tf.reduce_max(a, axis=1))
print(tf.reduce_min(a, axis=1))
print('均值：', tf.reduce_mean(a, axis=1))
print(tf.reduce_sum(a, axis=1))

print('---------get index--------')
# B=tf.constant([[2,20,30,3,6],[3,11,16,1,8],[14,45,23,5,27]])
# tf.math.argmax(B,0)
# [2, 2, 0, 2, 2] 比较的是每一列，第一列最大值是14，一维数组index为2， 第二列最大值是45，一维数组index为2, 以此类推....
a_random = tf.cast(tf.random.normal([2,5,10], mean=10, stddev=5), dtype=tf.int32)
print(a_random)
print(tf.argmax(a_random, axis=1))
print(tf.argmin(a_random, axis=1))

b = tf.constant([True, False, True, True])
print(tf.reduce_all(b))  # 逻辑与
print(tf.reduce_any(b))  # 逻辑或

print('-------范数---------')
x = tf.ones([2, 2])
print(tf.norm(x, ord=1))
print(tf.norm(x, ord=2))
print(tf.norm(x, ord=np.inf))
