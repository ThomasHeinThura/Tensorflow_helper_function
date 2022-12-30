import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from datetime import datetime
import pandas as pd

tf.get_logger().setLevel('ERROR')
tf.set_seed = 42
epoch = 10
input_shape = (32, 32, 3)

# import data
(train_features,train_labels), (test_features, test_labels) = tf.keras.datasets.reuters.load_data()

# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    )


print (train_features.item(1))
print (tf.constant(tf.expand_dims(train_features, axis=1), dtype=tf.float32))

"""
train_dataset  _dataset} \n"
      f"Test : {valid_dataset}")

"""


