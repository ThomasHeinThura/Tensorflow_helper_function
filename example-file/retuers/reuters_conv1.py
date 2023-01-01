import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from datetime import datetime
import pandas as pd
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.set_seed = 42
epoch = 10
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 100

# import data
(train_features,train_labels), (test_features, test_labels) = tf.keras.datasets.reuters.load_data(num_words=max_vocab_length)

# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    )

# VECTORIZE function

def vectorize_sequences(sequences, dimension=max_vocab_length):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return tf.cast(results, dtype= tf.float32)

# Vectorize and Normalize train and test to tensors with 10k columns

train_features_tf = vectorize_sequences(train_features)
test_features_tf = vectorize_sequences(test_features)

print("train_features ", train_features_tf.shape)
print("test_Features ", test_features_tf.shape)

# ONE HOT ENCODER of the labels

one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_features_tf, one_hot_train_labels))
train_dataset =  train_dataset.shuffle(8982).batch(32).cache().prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_features_tf,one_hot_test_labels))
valid_dataset = valid_dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

# Set random seed and create embedding layer (new embedding layer for each model)
embedding_layers = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=5,
                                     embeddings_initializer="uniform",
                                     input_length = max_length,
                                     name="embedding_layers")

# Build a Bidirectional RNN in TensorFlow
inputs = layers.Input(shape=(max_vocab_length,))
x = embedding_layers(inputs)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x) # condense the output of our feature vector
outputs = layers.Dense(46, activation="softmax")(x)
model= tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")

# Compile
model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of our bidirectional model
model.summary()

start = datetime.now()
# Fit the model (takes longer because of the bidirectional layers)
model_history = model.fit(train_dataset,
                           epochs=5,
                           validation_data=valid_dataset,
                           callbacks=[early_stopping, reduce_lr])

end = datetime.now()
print(f"The time taken to fit the modle is {end - start}")
model.evaluate(valid_dataset)