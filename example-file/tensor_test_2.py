import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
from datetime import datetime

(x_train, y_train), (x_test,y_test) = keras.datasets.cifar10.load_data()

#checking images shape
x_train.shape, x_test.shape

#checking labels
y_train[:5]

#scaling image values between 0-1
x_train_scaled = x_train/255
x_test_scaled = x_test/255

#one hot encolding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype= 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

#Building model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(3000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Model summary and Evaluation
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training
start = datetime.now()
model.fit(x_train_scaled, y_train_encoded, batch_size=128, epochs=10, validation_split=0.1)
stop = datetime.now()
print("Time taken to execute:" + str(stop - start))
