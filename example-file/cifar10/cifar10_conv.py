"""
This model is basic simple model of cifar10 dataset 
get poor performane val_accuary 63% and val_loss 1.2 and take 11min50sec
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

train_features = tf.cast(train_images, dtype=tf.float32) / 255
test_features = tf.cast(test_images, dtype=tf.float32) / 255
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)


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
model.compile(loss="categorical_crossentropy", # if labels aren't one-hot use sparse (if labels are one-hot, drop sparse)
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00075),
              metrics=["accuracy"])

start = datetime.now()
# Fit model
print("Training model...")
model.fit(x=train_images,
          y=train_labels,
          epochs=50,
          validation_data=(test_images, test_labels))
end = datetime.now()
print(f'The time taken to train the model is {end-start}')
# Evaluate model 
print("Evaluating model...")
model.evaluate(test_images, test_labels)

# Save model to current working directory
#model.save("test_image_model.h5")