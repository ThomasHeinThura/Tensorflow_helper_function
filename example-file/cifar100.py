import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

tf.get_logger().setLevel('ERROR')
tf.set_seed = 42
epoch = 100
input_shape = (32, 32, 3)

# import data
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.cifar100.load_data()

train_features = tf.cast(train_features, dtype=tf.float16)
train_labels = tf.cast(train_labels, dtype=tf.float16)
test_features = tf.cast(test_features, dtype=tf.float16)
test_labels = tf.cast(test_labels, dtype=tf.float16)


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

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=15) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=5,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

# Setup input shape and base model, freezing the base model layers

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False
inputs = layers.Input(shape=input_shape, name="input_layer")
x = base_model(inputs, training=False) 
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x) 
x = layers.Dense(1000, activation="relu", name ="hidden_layer_1")(x)
outputs = layers.Dense(100, activation="softmax",name="output_layer")(x)      
model = tf.keras.Model(inputs, outputs) 

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()

start = datetime.now()
history_model = model.fit(train_dataset,
                          steps_per_epoch=len(train_dataset),
                          validation_data=valid_dataset,
                          validation_steps=int(0.1*len(valid_dataset)),
                          callbacks=[early_stopping, reduce_lr],
                          epochs=epoch) 
end = datetime.now()
print(f"The time taken to train the model is {end - start}")
# Evaluate model
model.evaluate(valid_dataset)