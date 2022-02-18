import tensorflow as tf

"""
实际应用场景：提取张量中所有正数的数据和索引
"""
x = tf.random.normal([3,3])
print(x)

# 生成mask
mask = x > 0
print(mask)

# 获取为True的索引坐标
indices = tf.where(mask)
print(indices)

# 根据坐标获取元素：gather_nd
gt_zero = tf.gather_nd(x, indices)
print(gt_zero)

# 或者直接boolean_mask
bool_zero = tf.boolean_mask(x, mask)
print(bool_zero)
print(tf.equal(gt_zero, bool_zero))