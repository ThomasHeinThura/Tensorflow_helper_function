"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
#   # This is for helping the exam and others things make more easy #   #
# 1. for import data #
# 1.1 import from local data
# 1.2 import from tf.keras.dataset
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
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 1. for import data #
# 1.1 import from local data
def unzip_data(filename):
    import zipfile
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
    import os 
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
# Visualize classnes from dir
def view_class_name_from_dir(path):
    import pathlib 
    import numpy as np
    """
    View an class name from import data folder
    :param path:
    :return print class_name:
    """
    data_dir = pathlib.Path(path)
    class_name = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return print(class_name)

# 1.2 import from tf.keras.dataset # Procedures
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
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
# 2. for visualize import data #
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
    import random
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
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
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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




# Create a function to implement a ModelCheckpoint callback with a specific filename 
def create_model_checkpoint(model_name, save_path="model_experiments"):
    import os
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







