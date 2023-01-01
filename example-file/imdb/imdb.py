import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.set_seed = 42
epoch = 10
input_shape = (32, 32, 3)

# import data
(train_features,train_labels), (test_features, test_labels) = tf.keras.datasets.imdb.load_data()

# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    ) 
