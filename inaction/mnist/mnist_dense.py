import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers, Sequential, optimizers

(x, y), (x_test, y_test) = datasets.mnist.load_data()
x = tf.keras.utils.normalize(x)

train_data = tf.data.Dataset.from_tensor_slices((x, y))


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)  # 转成整型张量
    y = tf.one_hot(y, depth=10)
    return x, y


train_data = train_data.shuffle(1000).batch(128).map(preprocess)

model = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10)
])
# 也可以add
# model.add(keras.layers.Dense(64, activation='relu'))

model.build(input_shape=[None, 784])
model.summary()

optimizer = optimizers.SGD(learning_rate=0.001)

log_dir = "E:\program\workspace\pycharm\/tensorflow2\/tensorboard"
summary_writer = tf.summary.create_file_writer(log_dir)

def train():
    for epoch in range(3):
        for step, (x, y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.losses.MSE(y, out)

                # model.trainable_variables来获取参数列表
                grads = tape.gradient(loss, model.trainable_variables)

                # 更新参数
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                with summary_writer.as_default():
                    if step % 100 == 0:
                        print(epoch, step, loss)
                        tf.summary.scalar('train-loss', float(loss), step=step)


if __name__ == '__main__':
    train()
