import tensorflow as tf

"""
tensorflow中，rank表示tensor的秩
数值上来说等于ndim属性
"""

a = tf.random.normal([3, 5, 10, 15])
print(a.ndim)
print(tf.shape(a), a.shape)
print(tf.rank(a))

"""
测试 one-hot
输出的one-host矩阵是输入矩阵rank+1，expand了1维
行是输入的行，列是指定的depth
"""
data = tf.reshape(tf.range(10), [1,10])
one_hot = tf.one_hot(data, 15)
print(one_hot)