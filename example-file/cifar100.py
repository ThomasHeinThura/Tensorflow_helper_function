import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

tf.get_logger().setLevel('ERROR')
tf.set_seed = 42

# import data
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.cifar100.load_data()

train_features = tf.cast(train_features, dtype=tf.float16)
train_labels = tf.cast(train_labels, dtype=tf.float16)
test_features = tf.cast(test_features, dtype=tf.float16)
test_labels = tf.cast(test_labels, dtype=tf.float16)

# Check the data shape
print(f" Train : {train_features.shape} {train_features.dtype} \n {train_labels.shape} {train_labels.dtype} \n {test_features.shape} {test_features.dtype} \n {test_labels.shape} {test_labels.dtype} ") 

# Preprocess the data
# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
print(train_dataset)


# Setup input shape and base model, freezing the base model layers
input_shape = (32, 32, 3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = layers.Input(shape=input_shape, name="input_layer")      # Create input layer
x = base_model(inputs, training=False)   # Give base_model inputs (after augmentation) and don't train it
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)     # Pool output features of base model
outputs = layers.Dense(100, activation="softmax",name="output_layer")(x)        # Put a dense layer on as the output (Dense is same number of outputs as classes)
model = tf.keras.Model(inputs, outputs)         # Make a model with inputs and outputs

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history_model = model.fit(train_dataset,
                          batch_size=128,
                          steps_per_epoch=0.1*len(train_dataset),
                          validation_data=valid_dataset,
                          validation_steps=0.1*len(valid_dataset),
                          epochs=10) 

# Evaluate model
model.evaluate(test_features,test_labels)

# Predcit the value

#