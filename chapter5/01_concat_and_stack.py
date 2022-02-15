import tensorflow as tf

# a,b分别代表一个班35个同学8门课的成绩
a = tf.random.normal([35, 8], dtype=tf.float32)
b = tf.random.normal([35, 8], dtype=tf.float32)

# 70个人8门课的成绩，concat不产生新维度
# axis表示拼接的维度
c = tf.concat([a, b], axis=0)
print(c.shape)

# 2个班35个同学8门课的成绩，stack产生新维度
# axis表示新维度的位置
d = tf.stack([a, b], axis=2)
print(d.shape)


