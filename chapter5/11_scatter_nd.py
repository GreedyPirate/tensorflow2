import tensorflow as tf

indices = [[0,0],[1,1], [2, 2],[3,3],[4,4]]
updates = tf.range(5)

"""
把一个tensor，按照indices，插入到一个新的tensor中(默认都是0)
"""
new = tf.scatter_nd(indices, updates, [5,5])
print(new)