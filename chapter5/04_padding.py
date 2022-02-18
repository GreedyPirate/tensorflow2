import tensorflow as tf
from tensorflow import keras

a = tf.reshape(tf.range(100), [2, 5, 10])
print(a)

# 注意：即使某个维度不pad，也要写成[0,0], 即paddings列表长度=a.ndim
# [3,4]的意思是左边pad 3个，右边pad 4个
b = tf.pad(a, paddings = [[0,0],[3,4], [1,1]], constant_values=1)
print(b)


# 实际应用：设置句子 词最大长度为100，不足的pad
total_words = 1000
max_len = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len,truncating='post',padding='post')
print(x_train.shape)
