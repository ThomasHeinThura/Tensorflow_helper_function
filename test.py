from 1_import_dataset import *

train_data, test_data, = import_from_tfds("food101")

print(train_data)