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
