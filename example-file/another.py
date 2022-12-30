# Import all required libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf

# Split MNIST Train and Test data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# Divide train set by 255
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255
#x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
# convert class vectors to binary class matrices
y_train, y_test = keras.utils.to_categorical(y_train, num_classes=10), keras.utils.to_categorical(y_test, num_classes=10)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
# Model summary and Evaluation
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
start = datetime.now()
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
stop = datetime.now()
print("Time taken to execute:" + str(stop - start))
model.evaluate(x_test, y_test)
