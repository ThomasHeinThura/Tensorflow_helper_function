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
    """ Import from dataset
    # Get TensorFlow Datasets
    import tensorflow_datasets as tfds
    # List available datasets
    datasets_list = tfds.list_builders() # get all available datasets in TFDS
    print("food101" in datasets_list) # is the dataset we're after available?

    # Load in the data (takes about 5-6 minutes in Google Colab)
    (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                                 split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                                 shuffle_files=True, # shuffle files on download?
                                                 as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                                 with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)
    """
