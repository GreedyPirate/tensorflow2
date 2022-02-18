import tensorflow as tf

x = tf.reshape(tf.range(100), [10, 10])

""""
切片就是一种特殊的gather！
哪为什么要有gather，因为更灵活， index可以随意指定
"""
a = x[::, 4:8]  # 第二个维度要4-8列
b = tf.gather(x, [4, 5, 6, 7], axis=1)  # 单独指定维度(axis) 和 要的index列表
print(tf.equal(a, b))
print(tf.reduce_sum(tf.cast(tf.equal(a, b), dtype=tf.int32)) == tf.size(a))

# 也可以乱序
c = tf.gather(x, [8, 4, 7, 1], axis=0)

"""
gather_nd和gather没半毛钱关系，垃圾API，狗都不用
"""
print(x)
print(x[1:3, 2:5])
# 取第0行第3列元素， 取第5行第7列元素 ....
# 所以indices参数是一个个坐标？
d = tf.gather_nd(x, [[0, 3], [5, 7], [9, 7], [8, 1]])
print(d)

