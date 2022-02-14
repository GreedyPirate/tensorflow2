import tensorflow as tf

pic = tf.random.uniform([4, 32, 32, 3], 0, 256, dtype=tf.int32)
# print(pic)

slice0 = pic[0][1][3][0]
print(slice0)
# 等同于
slice1 = pic[1, 10, 10, 1]
print(slice1)

x_range = tf.range(100)
x_range = tf.reshape(x_range, [10, 10])
print(x_range)

# 不包含end，获取的是1,2  step=-1表示逆序
slice2 = x_range[1:3, ::-1]
print('slice2\n', slice2)
print('skip 2\n', x_range[1:7:2])

y_range = tf.reshape(x_range, [2, 5, 10])
print(y_range)
# 对三个维度切片. 第一个维度所有，第二个维度1-2，第三个维度2-8并且step=2
print('slice multi: ', y_range[:, 1:3, 2:8:2])
