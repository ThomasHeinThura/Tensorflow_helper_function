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

