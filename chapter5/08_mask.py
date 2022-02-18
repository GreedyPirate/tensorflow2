import tensorflow as tf


x = tf.reshape(tf.range(60), [4, 3, 5])
print(x)
mask = [True, False,True, False]

"""
也是取数，但是按照bool值确定是否取值，看上去和gather差不多
但bool可以由一些条件生成，或许写代码更方便， 
"""
a = tf.boolean_mask(x, mask, axis=0)
print('a: \n', a)

mask = [
    [True, False,True],
    [True, False, False],
    [False, False,True],
    [False, True,True]
        ]
print(x)
b = tf.boolean_mask(x, mask)
print('b:\n', b)

"""
mask为一维矩阵，是对x第一维的掩码
mask为二维矩阵，shape必须是x的前两维=[4,3]，表示第一维里，每第二维的取值
mask = [    
    [[False, True,True, False, True], [False, True,True, False, True],[False, True,True, False, True]],
    [[False, True,True, False, True], [False, True,True, False, True], [False, True,True, False, True]],
    [[False, True,True, False, True], [False, True,True, False, True],[False, True,True, False, True]],
    [[False, True,True, False, True], [False, True,True, False, True],[False, True,True, False, True]]
]
"""