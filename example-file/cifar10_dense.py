"""
The model perform so poor(val_accuary 53% and val_loss 1.4 take 10min to train) but keep this file to grip 
the knowleages on building the basic model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
from datetime import datetime

input_shape = (32, 32, 3)

(train_features, train_labels), (test_features,test_labels) = keras.datasets.cifar10.load_data()

train_features = tf.cast(train_features, dtype=tf.float32) / 255
test_features = tf.cast(test_features, dtype=tf.float32) / 255
train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)


# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    ) 

# Preprocess the data
# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset =  train_dataset.shuffle(5000).batch(128).prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_features,test_labels))
valid_dataset = valid_dataset.batch(128).prefetch(tf.data.AUTOTUNE)
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")

#Building model

model = keras.Sequential([
    keras.layers.Input(shape=input_shape, name="input_layer"),
    keras.layers.Flatten(),
    keras.layers.Dense(3000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Model summary and Evaluation
model.summary()
model.compile(loss="categorical_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(learning_rate= 0.00075), 
              metrics=["accuracy"])

# Training
start = datetime.now()
history_model = model.fit(train_dataset,
                          steps_per_epoch=len(train_dataset),
                          validation_data=valid_dataset,
                          validation_steps=int(0.1*len(valid_dataset)),
                          epochs=25) 
stop = datetime.now()
print("Time taken to execute:" + str(stop - start))
model.evaluate(valid_dataset)
