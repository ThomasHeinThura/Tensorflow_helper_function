import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from datetime import datetime
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.set_seed = 42
epoch = 10

# import data
(train_features,train_labels), (test_features, test_labels) = tf.keras.datasets.reuters.load_data()

# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    )

train_features_np = np.array(train_features)

print(train_features_np.shape, train_features_np.dtype)
print(len(train_features))

print(train_features[1])

tensor_test = tf.cast(train_features[1], dtype=tf.float32)

print(tensor_test)

for i in range(5):
    tensor_ds = []
    print(train_features[i])
    j = tf.cast(train_features[i],dtype=tf.float32)
    tensor_ds.append([i,j])
print(tensor_ds.shape)






