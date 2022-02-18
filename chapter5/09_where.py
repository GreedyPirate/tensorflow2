import tensorflow as tf

a = tf.ones([3,3])
b = tf.zeros([3,3])
conds = tf.constant([[True,False,False],[False,True,False],[True,True,False]])

"""
结果：
[[1. 0. 0.]
 [0. 1. 0.]
 [1. 1. 0.]]
where的意思是：
根据conds索引处的值，如果为true，取a； 如果为false，取b
"""
c = tf.where(conds, a, b)
print(c)

# 不传a，b，返回所有为True的坐标
print(tf.where(conds))