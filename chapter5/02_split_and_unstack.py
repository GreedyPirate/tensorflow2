import tensorflow as tf

# split和unstack，即concat和stack的逆操作
a = tf.random.normal([10, 35, 8], dtype=tf.float32)
b = tf.random.normal([10, 35, 8], dtype=tf.float32)

# axis：要切割的维度，返回list
c_list = tf.split(a, num_or_size_splits=2, axis=0)
print(len(c_list), c_list[0].shape)

# 切割第一个维度(长度为10)，并且按4+3+3=10的分法
c2_list = tf.split(a, num_or_size_splits=[4, 3, 3], axis=0)
print(len(c2_list), c2_list[0].shape)

# 虽然是stack的逆操作，但是只能切割成1份，也很好理解，要降维了，只能按1份份切割
d_list = tf.unstack(b, axis=0)
print(len(d_list), d_list[0].shape)
