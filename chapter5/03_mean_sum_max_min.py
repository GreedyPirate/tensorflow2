import tensorflow as tf
import numpy as np

a = tf.reshape(tf.range(4), [2, 2])
print(a)

print(tf.norm(a, ord=1))
print(tf.norm(a, ord=2))
print(tf.norm(a, ord=np.inf))
