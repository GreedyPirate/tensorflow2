import tensorflow as tf

# 矩阵乘法， [m, k] @ [k, n] = [m, n]

# 1. tf矩阵乘，只要求最后两个维度，符合上述规律
# 2. 除了最后两个维度，其他任意维度上，都需要“相等”或“有一个是1”
# 1, 10, 5, 2
# 2, 10, 2, 10
# [5,2]@[2,10] = [5, 10]
# 前面的维度，axis=0，其中一个是1；axis=2，两个都是10
matrix_A = tf.reshape(tf.range(100), [1, 10, 5, 2])
matrix_B = tf.reshape(tf.range(400), [2, 10, 2, 10])
print((matrix_A@matrix_B).shape)

# axis=[0,1]相等
# matrix_A = tf.reshape(tf.range(200), [2, 10, 5, 2])
# matrix_B = tf.reshape(tf.range(400), [2, 10, 2, 10])

# matrix_C会自动expand为[1,10, 5, 2]
matrix_C = tf.reshape(tf.range(100), [10, 5, 2])
matrix_D = tf.reshape(tf.range(400), [2, 10, 2, 10])
print((matrix_C@matrix_D).shape)

