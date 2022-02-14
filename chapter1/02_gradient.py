import tensorflow as tf;
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(2.5)
b = tf.constant(3.)
x = tf.constant(4.)
c = tf.Variable(2.)

# 求导失败返回None
# persistent可以求导多次
# constant必须watch才能求导，variable不用，但把watch_accessed_variables=False，则所有都要watch才行
with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
    tape.watch([x, a])
    y = a * x ** 2 + 2 * b ** 3 + tf.pow(c, 2)

    [dy_dx] = tape.gradient(y, [x])
    print(dy_dx)

    [dy_da] = tape.gradient(y, [a])
    print(dy_da)

    [dy_dc] = tape.gradient(y, [c])
    print(dy_dc)
