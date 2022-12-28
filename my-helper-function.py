# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
#   # This is for helping the exam and others things make more easy #   #
# 1. for import data #
# 1.1 import from local data
# 1.2 improt from cvs
# 1.3 import from online data
# 1.4 import from tf.keras.dataset
#
# 2. for visualize import data #
# 2.1 visualize plot data
# 2.2 visualize picture
# 2.3 visualize words and panda
#
# 3. for preparation data (normalize and add to pipeline)
# 3.1 data work thorugh for CNN
# 3.2 datagen for CNN
# 3.3 token and embedd for NLP 
# 3.4 windows and horizon for timeseries
# 3.5 data for classification and regression 
#
# 4. Fit the model and make sure to remember history and callbacks 
# 4.1 early stopping callbacks
# 4.2 plateua for learning rate reducing
# 4.3 save the best perfromance models 
#
# 5. visualize the model and plot prediction and matrix
# 5.1 model visualization
# 5.2 plot prediction 
# 5.3 confusion matrix 
# 5.4 the most common mistakes (ture labels and prediction show) for CNN and NLP
# 5.5 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# import for python 
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import zipfile
import os
import random

"""
# Turn off all warnings except for errors
tf.get_logger().setLevel('ERROR')
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 1. for import data #
# 1.1 import from local data
def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
# data work thorugh dirpath for CNN
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
  
# 1.2 improt from cvs
""" import from cvs:
import pandas as pd
from tensorflow import tf.datapath = ‘path/to/your/csv/file’
df = pd.read_csv(path)
dataset = Dataset.from_tensor_slices(dict(df))
"""

# 1.3 import from online data
def show_methods_for_import_online():
  """
  This is the example procedure to do in cvs data
  import pandas as pd
  # Read in the insurance dataset
  insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
  # Check out the insurance dataset
  insurance.head()
  # Turn all categories into numbers
  insurance_one_hot = pd.get_dummies(insurance)
  insurance_one_hot.head() # view the converted columns
  # Create X & y values
  X = insurance_one_hot.drop("charges", axis=1)
  y = insurance_one_hot["charges"]
  # Create training and test sets
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=42) 


  from sklearn.compose import make_column_transformer
  from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

  # Create column transformer (this will help us normalize/preprocess our data)
  ct = make_column_transformer(
      (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
      (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
  )

  # Create X & y
  X = insurance.drop("charges", axis=1)
  y = insurance["charges"]

  # Build our train and test sets (use random state to ensure same split as before)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Fit column transformer on the training data only (doing so on test data would result in data leakage)
  ct.fit(X_train)

  # Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)
  X_train_normal = ct.transform(X_train)
  X_test_normal = ct.transform(X_test)

  """

# 1.4 import from tf.keras.dataset
def show_methods_for_import_dataset():
  """
  import tensorflow as tf
  from tensorflow.keras.datasets import fashion_mnist

  # The data has already been sorted into training and test sets for us
  (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

  # Viewing the single example for fashion_mnist
  # Plot a single example
  import matplotlib.pyplot as plt
  plt.imshow(train_data[7]);

  # Plot an example image and its label
  plt.imshow(train_data[17], cmap=plt.cm.binary) # change the colours to black & white
  plt.title(class_names[train_labels[17]]);
  """

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 2. for visualize import data #
# 2.1 visualize plot data


# 2.2 Visualize classnes from dir
def view_class_name_from_dir(path):
    """
    View an class name from import data folder
    :param path:
    :return print class_name:
    """
    data_dir = pathlib.Path(path)
    class_name = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return print(class_name)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 2.3 visualize picture
# Plot multiple random images of fashion MNIST
"""
import random
plt.figure(figsize=(7, 7))
for i in range(4):
  ax = plt.subplot(2, 2, i + 1)
  rand_index = random.choice(range(len(train_data)))
  plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
  plt.title(class_names[train_labels[rand_index]])
  plt.axis(False)

"""
def view_random_image(target_dir, target_class):
    """
    View an random image from import data
    :param target_dir:
    :param target_class:
    :return img:
    """
    # Setup target directory (we'll view images from here)
    target_folder = target_dir+target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");
    print(f"Image shape: {img.shape}") # show the shape of the image

    return img

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# 2.4 visualize words and pandas
"""
# Download data (same as from Kaggle)
!wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
unzip_data("nlp_getting_started.zip")

# Turn .csv files into pandas DataFrame's
import pandas as pd
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()

# Let's visualize some random training examples
import random
random_index = random.randint(0, len(train_df)-5) # create random indexes not higher than the total number of samples
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 3. for preparation data (normalize and add to pipeline)
# 3.1 Train test split
"""
from sklearn.model_selection import train_test_split

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, # dedicate 10% of samples to validation set
                                                                            random_state=42) # random state for reproducibility
"""
# 3.2 datagen for CNN
# preparation CNN data (first way)
def reshape_image_from_dir_to_train(train_dir,test_dir):
  # Set the seed
  tf.random.set_seed(42)

  # Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
  train_datagen = ImageDataGenerator(rescale=1./255)
  valid_datagen = ImageDataGenerator(rescale=1./255)

  # Import data from directories and turn it into batches
  train_data = train_datagen.flow_from_directory(train_dir,
                                                 batch_size=32, # number of images to process at a time 
                                                 target_size=(224, 224), # convert all images to be 224 x 224
                                                 class_mode="binary", # type of problem we're working on
                                                 seed=42,
                                                 shuffle=True)

  valid_data = valid_datagen.flow_from_directory(test_dir,
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode="binary",
                                                 seed=42,
                                                 shuffle=True)

  return train_data, valid_data

def reshape_augmented_image_from_dir_to_train(train_dir,test_dir):
  # Set the seed
  tf.random.set_seed(42)

  # Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)

  train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                               rotation_range=20, # rotate the image slightly between 0 and 20 degrees (note: this is an int not a float)
                                               shear_range=0.2,  # shear the image
                                               zoom_range=0.2,  # zoom into the image
                                               width_shift_range=0.2,  # shift the image width ways
                                               height_shift_range=0.2,  # shift the image height ways
                                               horizontal_flip=True)  # flip the image on the horizontal axis

  valid_datagen = ImageDataGenerator(rescale=1./255)

  # Import data from directories and turn it into batches
  train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                 batch_size=32, # number of images to process at a time
                                                 target_size=(224, 224), # convert all images to be 224 x 224
                                                 class_mode="binary", # type of problem we're working on
                                                 seed=42,
                                                 shuffle= True)

  valid_data = valid_datagen.flow_from_directory(test_dir,
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode="binary",
                                                 seed=42,
                                                 shuffle=True)
  return train_data_augmented, valid_data

# preparation CNN data (second way)
def preprocess_image_using_keras(train_dir, test_dir):
    # Create data inputs
    import tensorflow as tf
    IMG_SIZE = (224, 224)  # define image size
    train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                                image_size=IMG_SIZE,
                                                                                label_mode="categorical",
                                                                                # what type are the labels?
                                                                                batch_size=32)  # batch_size is 32 by default, this is generally a good number
    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                               image_size=IMG_SIZE,
                                                                               label_mode="categorical")
    return train_data, test_data
# The second way is better and Data-augmented-layers is needed.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 3.3 data pipe line for best performace
"""
# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 3.4 tokenization(text vectorization) and embedding for NLP 
"""
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# Note: in TensorFlow 2.6+, you no longer need "layers.experimental.preprocessing"
# you can use: "tf.keras.layers.TextVectorization", see https://github.com/tensorflow/tensorflow/releases/tag/v2.6.0 for more

# Use the default TextVectorization variables
text_vectorizer = TextVectorization(max_tokens=None, # how many words in the vocabulary (all of the different words in your text)
                                    standardize="lower_and_strip_punctuation", # how to process text
                                    split="whitespace", # how to split tokens
                                    ngrams=None, # create groups of n-words?
                                    output_mode="int", # how to map tokens to numbers
                                    output_sequence_length=None) # how long should the output sequence of tokens be?
                                    # pad_to_max_tokens=True) # Not valid if using max_tokens=None

# Setup text vectorization with custom variables
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
"""

"""
# Choose a random sentence from the training dataset and tokenize it
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nVectorized version:")
text_vectorizer([random_sentence])
"""
"""
# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = words_in_vocab[-5:] # least common tokens
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}") 
print(f"Bottom 5 least common words: {bottom_5_words}")
"""

"""
tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1") 

embedding
"""
"""
# Get a random sentence from training set
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nEmbedded version:")

# Embed the random sentence (turn it into numerical representation)
sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 3.5 windows and horizon for timeseries
""" taken form time series py"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 3.6 data for classification and regression (min_max and normalization)
""" search from scikit learn min_max_scaler and train test split random
# Wrong way to make train/test sets for time series
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(timesteps, # dates
                                                    prices, # prices
                                                    test_size=0.2,
                                                    random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape 


"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 4. Fit the model and make sure to remember history and callbacks 
# 4.1 early stopping callbacks (fix file from cnn_advence)
"""
# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 4.2 plateua for learning rate reducing (fix file from cnn_advence)
"""
# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 4.3 save the best perfromance models aka modelcheckpoint(fix file from cnn_advence)
"""
# Create TensorBoard callback (already have "create_tensorboard_callback()" from a previous notebook)
from helper_functions import create_tensorboard_callback

# Create ModelCheckpoint callback to save model's progress
checkpoint_path = "model_checkpoints/cp.ckpt" # saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor="val_accuracy",/monitor='val_loss' # save the model weights with best validation accuracy
                                                      save_best_only=True, # only save the best weights
                                                      save_weights_only=True, # only save model weights (not whole model)
                                                      verbose=0) # don't print out whether or not model is being saved 
"""
import os

# Create a function to implement a ModelCheckpoint callback with a specific filename 
def create_model_checkpoint(model_name, save_path="model_experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), # create filepath to save model
                                            verbose=0, # only output a limited amount of text
                                            save_best_only=True) # save only the best model to file
                                            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 4.4 Creat tensorboard and can show history of models
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

#4.5 mixed precision training
# Turn on mixed precision training (that is to train with faster)
# if you need check tensorflow.keras.mixed_precision
"""
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 

mixed_precision.global_policy() # should output "mixed_float16" (if your GPU is compatible with mixed precision)

example build model for mix precision you need to add dtype=tf.float32 to use mix precsion
from tensorflow.keras import layers

# Create base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False # freeze base model layers

# Create Functional model 
inputs = layers.Input(shape=input_shape, name="input_layer")
# Note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
# x = layers.Rescaling(1./255)(x)
x = base_model(inputs, training=False) # set base_model to inference mode only
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dense(len(class_names))(x) # want one output neuron per class 
# Separate activation of output layer so we can output float32 activations
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) 
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Check the dtype_policy attributes of layers in our model
for layer in model.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # Check the dtype policy of layers

# Check the layers in the base model and see what dtype policy they're using
for layer in model.layers[1].layers[:20]: # only check the first 20 layers to save output space
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 5. visualize the model and plot prediction and matrix
# 5.1 model visualization
"""  model.summary and plot_model(plot_model only work in jupyter)
model.summary()

import tensorflow as tf 

input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
x = tf.keras.layers.Embedding(
    output_dim=512, input_dim=10000, input_length=100)(input)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
model = tf.keras.Model(inputs=[input], outputs=[output])
model.summary()
tf.keras.utils.plot_model(model)


"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 5.2 plot prediction 
# Create a function for plotting a random image along with its prediction

def plot_predictions(train_data, train_labels, test_data, test_labels,predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend();

def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
     1. CS231n - https://cs231n.github.io/neural-networks-case-study/
     2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[
        xx.ravel(), yy.ravel()]  # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[
        -1] > 1:  # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Plot the validation and training data separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

# prepare to predict certain image
def reshape_image_to_predict(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor with normalization and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

# Make a function to predict on images(CNN model) and plot them (works with multi-class)
def predict_reshaped_image_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = reshape_image_for_predict(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);

def show_preds_image(model, class_names, test_dir):
    # Make preds on a series of random images
    import os
    import random

    plt.figure(figsize=(17, 10))
    for i in range(3):
        # Choose a random image from a random class
        #class_names = test_data.class_names
        class_name = random.choice(class_names)
        filename = random.choice(os.listdir(test_dir + "/" + class_name))
        filepath = test_dir + class_name + "/" + filename

        # Load the image and make predictions
        img = reshape_image_for_predict(filepath, scale=False)  # don't scale images for EfficientNet predictions
        pred_prob = model.predict(tf.expand_dims(img, axis=0))  # model accepts tensors of shape [None, 224, 224, 3]
        pred_class = class_names[pred_prob.argmax()]  # find the predicted class

        # Plot the image(s)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img / 255.)
        if class_name == pred_class:  # Change the color of text based on whether prediction is right or wrong
            title_color = "g"
        else:
            title_color = "r"
        plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
        plt.axis(False);

def predict_random_image(model, images, true_labels, classes): # predict for fashion mninst
    """Picks a random image, plots it and labels it with a predicted and truth label.

    Args:
      model: a trained model (trained on data similar to what's in images).
      images: a set of random images (in tensor form).
      true_labels: array of ground truth labels for images.
      classes: array of class names for images.

    Returns:
      A plot of a random image from `images` with a predicted class label from `model`
      as well as the truth class label from `true_labels`.
    """
    # Setup random integer
    i = random.randint(0, len(images))

    # Create predictions and targets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))  # have to reshape to get into right size for model
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # Plot the target image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Change the color of the titles depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Add xlabel information (prediction/true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100 * tf.reduce_max(pred_probs),
                                                     true_label),
               color=color)  # set the color to green or red

"""  # Plot multiple random images of fashion MNIST
  import random
  plt.figure(figsize=(7, 7))
  for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)
"""

# Compare two history
def compare_two_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show() 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 5.3 confusion matrix 
# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# Function to evaluate: accuracy, precision, recall, f1-score
def calculate_accuracy_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

# 5.4 the most common mistakes (ture labels and prediction show) for CNN and NLP
# fix file from cnn.py and commit

# 5.5 