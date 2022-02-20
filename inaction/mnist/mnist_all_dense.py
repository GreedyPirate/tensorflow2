import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras import layers, Sequential, optimizers

(x, y), (x_test, y_test) = datasets.mnist.load_data()
x = tf.keras.utils.normalize(x)

train_data = tf.data.Dataset.from_tensor_slices((x, y))
test_data  = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)  # 转成整型张量
    y = tf.one_hot(y, depth=10)
    return x, y


train_data = train_data.shuffle(1000).batch(128).map(preprocess)
test_data = test_data.shuffle(1000).batch(128).map(preprocess)

model = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 784])
model.summary()

# optimizer = optimizers.SGD(learning_rate=0.001)

model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=tf.losses.MSE,
              metrics=['accuracy'])

history = model.fit(train_data, epochs=3, validation_data=test_data, validation_freq=2)
print('history\n', history.history)

metrics = model.evaluate(test_data)
print('metrics\n', metrics)

# 只保存权重， load需要重新定义网络结构，然后load_weights
model.save_weights('E:\program\workspace\pycharm\/tensorflow2\modelwight\mnist\mnist-weights.ckpt')
# model.load_weights('mnist-weights.ckpt')

model.save('E:\program\workspace\pycharm\/tensorflow2\modelfile\mnist\model.h5')
# keras.models.load_model('model.h5')

# saved-model 平台无关性
tf.saved_model.save(model, 'E:\program\workspace\pycharm\/tensorflow2\modelfile\mmnist/')
# tf.saved_model.load('E:\program\workspace\pycharm\/tensorflow2\modelfile')

del model