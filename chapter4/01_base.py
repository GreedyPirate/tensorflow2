import tensorflow as tf
import numpy as np

a = tf.constant(1.)
print(type(a), tf.is_tensor(a))

vector = tf.constant([1., 2., 3.])
print(vector.numpy())

matrix = tf.constant([[1., 2.], [3., 4.]])
print(matrix.numpy())

d3 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(d3.numpy())

str = tf.constant('Hello,tensor')
lowerStr = tf.strings.lower(str)
splitStr = tf.strings.split(str, sep=',')
print(str.numpy(), lowerStr.numpy(), splitStr.numpy())

# bool类型的标量和向量
flag = tf.constant(True)
tf.constant([True, False])
# 只能用 == 比较
print('yes' if flag == False else 'no')

# 精度转换
d_float = tf.constant(np.pi, dtype=tf.float16)
print(d_float)
if (d_float.dtype != tf.float64):
    d_float = tf.cast(d_float, tf.float64)
print(d_float)

## bool和int之间的转换
b_arr = tf.constant([True, False])
i_arr = tf.cast(b_arr, tf.int8)
print(i_arr)

# 非0为true
i_arr = tf.constant([-1, 2, 0, 3, 0])
b_arr = tf.cast(i_arr, tf.bool)
print(i_arr)
print(b_arr)
