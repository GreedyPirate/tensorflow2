import tensorflow as tf

"""
讨论tensorflow中如果实现取值范围:
x ∈ [5, +∞)
x ∈ [10, 50]
...
"""

x = tf.range(100)

# maximum函数：返回2个参数的最大值，x中所有 < 5的，都会覆盖为5
x = tf.maximum(x, 5)
print(x)

# 同理
x = tf.minimum(x, 80)
print(x)

# 上面二者的集合体，所有元素满足[10, 30]
y = tf.clip_by_value(x, 10, 30)
print(y)