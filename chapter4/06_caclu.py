import tensorflow as tf

# broadcast的限制，shape从右往左比较，要么相等，要么一个是1
a_range = tf.range(start=1, limit=51)
b_range = tf.range(10)
a_range = tf.reshape(a_range, [5, 5, 2])
b_range = tf.reshape(b_range, [5, 1, 2])

c_range = tf.add(b_range, a_range)

a_range = tf.reshape(a_range, [5, 10, 1])
b_range = tf.reshape(b_range, [5, 1, 2])
d_range = b_range + a_range
print(d_range)

# tf重载了运算符，所以+可以替代tf.add
print(b_range - a_range)
print(b_range * a_range)
print(b_range / a_range)
print(b_range // a_range)  # 整除
print(b_range % a_range)  # 取余
