from import_dataset import *

train_data, train_label, test_data, test_labels = import_from_tfds("food101")

print(train_data)