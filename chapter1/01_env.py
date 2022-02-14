import tensorflow as tf;

print(tf.__version__)
print(tf.test.is_gpu_available())

# TensorFlow 在运行时，默认会占用所有 GPU 显存资源 改为增长式占用模式
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


with tf.device("/gpu:0"):
    a = tf.constant(2.)
    b = tf.constant(3.)
    print(a * b)

