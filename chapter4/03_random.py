import tensorflow as tf

f_rand = tf.random.normal([2,2], mean=1.0, stddev=0.5, dtype=tf.float32)
print(f_rand)

f_uni = tf.random.uniform([3,3], 0, 10, dtype=tf.float32)
print(f_uni)

# tf.int16报错
range_10 = tf.range(start=10, limit=200, delta=10, dtype=tf.int32, name="range-10")
print(range_10)