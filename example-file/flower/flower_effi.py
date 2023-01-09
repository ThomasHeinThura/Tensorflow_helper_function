"""
The model performance : 
val_accuary : 71.03%
val_loss : 0.7399
time :  18min15sec
epoch : 20
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL
import PIL.Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
import tensorflow_datasets as tfds
import pathlib
from datetime import datetime
from tensorflow.keras import layers

batch_size = 32
img_height = 128
img_width = 128
AUTOTUNE = tf.data.AUTOTUNE
input_shape = (img_height,img_width, 3)
num_classes = 5
epoch = 20

# Import Data
flower_dir = '/home/hanlinn/tensorflow_datasets/flowers/'

train_ds = tf.keras.utils.image_dataset_from_directory(
  flower_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  flower_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print(train_ds)
print(val_ds)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)
from tensorflow.keras.layers.experimental import preprocessing

# Create a data augmentation stage with horizontal flipping, rotations,
data_augmentation = tf.keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name ="data_augmentation")

inputs = layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)
x = layers.Conv2D(32, kernel_size=3, padding="same", activation="elu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(64, kernel_size=3, padding="same", activation="elu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128, kernel_size=3, padding="same" ,activation="elu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(256, kernel_size=3, padding="same" ,activation="elu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(512, kernel_size=3, padding="same" ,activation="elu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(128, activation="elu", name="Dense_1")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="elu", name="Dense_2")(x)
outputs = layers.Dense(num_classes, activation="softmax",name="output_layer")(x)      
model = tf.keras.Model(inputs, outputs) 


model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

model.summary()

start = datetime.now()
history_model = model.fit(train_ds,
                          steps_per_epoch=len(train_ds),
                          validation_data=val_ds,
                          validation_steps=int(0.25*len(val_ds)),
                          callbacks=[early_stopping, reduce_lr],
                          epochs=epoch) 
end = datetime.now()


print(f"The time taken to train the model is {end - start}")
# Evaluate model
model.evaluate(val_ds)

