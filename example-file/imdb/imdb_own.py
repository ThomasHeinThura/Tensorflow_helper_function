"""
The base model performance 
val_accuary : 8 in 10 epochs
val_loss : 
time : 
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras import layers 
from tensorflow.keras.layers import TextVectorization 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbos
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

tf.set_seed = 42
epoch = 10

# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

#Tokenization and embedding layers
# Set random seed and create embedding layer (new embedding layer for each model)
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
embedding_layers = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=5,
                                     embeddings_initializer="uniform",
                                     input_length = max_length,
                                     name="embedding_layers")

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(text_vectorizer)
model.add(embedding_layers)
#model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),)
model.add(tf.keras.layers.LSTM(8,))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


start = datetime.now()
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=epoch,
                    validation_data=validation_data.batch(512),
                    verbose=1)

end = datetime.now()

print(f"The time taken to train the model is :{end - start}")
results = model.evaluate(test_data.batch(512))