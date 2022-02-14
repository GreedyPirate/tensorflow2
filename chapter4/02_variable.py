import tensorflow as tf
import numpy as np;

#  Variable
# 一种专门的数据类型来支持梯度信息的记录
w = tf.Variable([1.], name="w")
print(w.name, w.numpy(), w.shape, w.trainable)

# 转为了 float32
p_list = tf.convert_to_tensor([1., 2.])
print(p_list)

# 转为了 float64
np_arr = np.array([3., 4.])
np_tensor = tf.convert_to_tensor(np_arr)
print(np_tensor)

# 标量
print(tf.zeros([]), tf.ones([]))
# 向量
print(tf.zeros([2]), tf.ones([3]))
# 矩阵
print(tf.zeros([2,2]), tf.ones([3,3]))

a = tf.zeros([2,3,4])
# 和a shape一样的矩阵， ones_like
print(tf.zeros_like(a))

# 自定义shape和value
print(tf.fill([2,2], 10))

