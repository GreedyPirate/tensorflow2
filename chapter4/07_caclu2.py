import tensorflow as tf

a = tf.constant([1, 2, 3, 4])
print(a.shape)
a = tf.reshape(a, [2, 2])
b = tf.fill([2, 2], 2)

c = a ** 2  # =tf.pow
print(c)

d = 2 ** a  # 2的n次方
print(d)

e = tf.exp(1.)  # e的1次方
print(e)

c = tf.cast(c, dtype=tf.float32)  # 先转float
print(c ** 0.5)  # c的1/2次方，开方运算


