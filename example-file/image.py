"""
This script should train a TensorFlow model on the fashion MNIST dataset to ~90% test accuracy.

It'll save the model to the current directory using the ".h5" extension.

You can use it to test if your local machine is fast enough to complete the
TensorFlow Developer Certification.

If this script runs in under 5-10 minutes through PyCharm, you're good to go.

The models/datasets in the exam are similar to the ones used in this script.
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers
from datetime import datetime

input_shape = (32, 32, 3)

# Check version of TensorFlow (exam requires a certain version)
# See for version: https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf 
print(tf.__version__)
tf.get_logger().setLevel('ERROR')
# Get data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize images (get values between 0 & 1)
train_images, test_images = train_images / 255.0, test_images / 255.0 

# Check shape of input data
# print(train_images.shape)
# print(train_labels.shape)

# Build model
model = tf.keras.Sequential([
    # Reshape inputs to be compatible with Conv2D layer
    #layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    layers.Input(shape=input_shape, name="input_layer"),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.Flatten(), # flatten outputs of final Conv layer to be suited for final Dense layer
    layers.Dense(10, activation="softmax")
])

# Compile model 
model.compile(loss="sparse_categorical_crossentropy", # if labels aren't one-hot use sparse (if labels are one-hot, drop sparse)
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

start = datetime.now()
# Fit model
print("Training model...")
model.fit(x=train_images,
          y=train_labels,
          epochs=25,
          validation_data=(test_images, test_labels))
end = datetime.now()
print(f'The time taken to train the model is {end-start}')
# Evaluate model 
print("Evaluating model...")
model.evaluate(test_images, test_labels)

# Save model to current working directory
#model.save("test_image_model.h5")
