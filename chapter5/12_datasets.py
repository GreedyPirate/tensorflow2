import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets

# 1. 加载数据
(x, y), (x_test, y_test) = datasets.mnist.load_data()
# 2. 归一化，为什么在这里不用map_fn预处理，因为normalize接收的参数ndarry类型
x = tf.keras.utils.normalize(x)
print(type(x), x.shape, '/', type(y), y.shape)

train_data = tf.data.Dataset.from_tensor_slices((x, y))
print(train_data)

# reduce_max兼任np array
print(type(x), tf.reduce_max(x))
print(x.min(), x.max())

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)  # 转成整型张量
    y = tf.one_hot(y, depth=10)
    return x,y

train_data = train_data.take(4096).shuffle(1000).batch(1024).map(preprocess)

"""
迭代出来的element就是一个batch里的(x,y)
element[0].shape = [batchSize, x.shape]
element[1].shape = [batchSize, y.shape]
"""
iterator = iter(train_data)
element = next(iterator)

print('iterator element:\n', type(element), len(element), element)


