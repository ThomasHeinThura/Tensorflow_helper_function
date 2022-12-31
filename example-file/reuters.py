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

for i in range(10):
    print(train_features[i])
    tf_train_features = tf.reshape(i ,train_features[i])
    print(tf_train_features)
    #print(train_features[i].shape)
    #print(train_features[i].dtype)





