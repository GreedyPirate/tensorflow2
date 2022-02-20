import tensorflow as tf
from tensorflow import keras

# 从外面加载数据
dataset_path = keras.utils.get_file("auto-mpg.data",
"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/autompg.data")