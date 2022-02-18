import tensorflow as tf
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(type(x_train))  # ndarray
"""
转换为tensorflow TensorSliceDataset对象
"""
tf_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print(type(tf_train))
"""
防止每次训练时数据按固定顺序产生
"""
shuffled = tf_train.shuffle(10000)
"""
map_fn， 就是java里的map，主要做数据预处理，参数是一个函数
"""
shuffled.map()




def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10, on_value=10, off_value=-1)
