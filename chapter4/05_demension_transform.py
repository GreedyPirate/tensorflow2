import tensorflow as tf
import numpy as np

x_range = tf.range(100)
# 最左边：大维度， 最右边：小维度
# -1 表示自动推算最后一个维度，所以只能有一个-1，同时已知的维度要能被个数整除，例如[4,8,-1]就不能被100整除
x_range = tf.reshape(x_range, [4, 5, -1])
# 获取维度数
print(x_range.ndim)

# 增加维度
b_range = tf.range(20)
b_range = tf.reshape(b_range, [2, 10])
# 只能增加长度为1的维度
# 怎么理解axis参数，新维度的位置！
# 新维度位置为2, [2,10,1]
add_dim = tf.expand_dims(b_range, axis=2)
print(add_dim.ndim, add_dim.shape)
# 新维度位置为1, [2,1,10,1]
add_dim = tf.expand_dims(add_dim, axis=1)
print(add_dim.ndim, add_dim.shape)

# 删除维度
x_range = tf.reshape(x_range, [2, 5, 1, 1, -1])
print(x_range.ndim, x_range.shape)
# 默认删除所有长度为1的维度
rm_dim = tf.squeeze(x_range)
print('remove all ', rm_dim.shape)
rm_dim = tf.expand_dims(rm_dim, axis=1)
print(rm_dim.shape)
rm_dim = tf.squeeze(rm_dim, axis=1)
print('remove rm_dim[0]', rm_dim.shape)

# 能删除长度!=1的维度吗，不能！  Can not squeeze dim[1], expected a dimension of 1, got
# rm_dim = tf.squeeze(rm_dim, axis=1)

# 交换维度
# 目前是[2,5,10], 维度索引=[0,1,2]， 设置为[2,1,0]变为[10, 5, 2]
trans_dim = tf.transpose(rm_dim, [2, 1, 0])
print(trans_dim)

# 复制
# 以下步骤，tf自动完成，expand+tile=broadcast
print('------------tile---------------')
b_tile = tf.constant([0, 1, 2])
b_tile = tf.expand_dims(b_tile, axis=0)
print(b_tile)
b_tile = tf.tile(b_tile, [2, 1])
print(b_tile)

# broadcast
# broadcast效果上=tile，但是它是逻辑上改变shape，只有到正在执行时才复制
# tile时立即创建一个新的tensor，有IO操作
x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.random.normal([3])
y = x@w + b  # +b自动完成broadcast，  tf.broadcast_to(b, [2,3])
print(y)
