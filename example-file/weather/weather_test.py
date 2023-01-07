import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from windows import WindowGenerator
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# import dataset
df = pd.read_csv(csv_path)
#Remove datetime
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
# fix error
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0
# fix error
max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
#Change wide direction degree to sin and cos
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

# change time to sin and cos
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

#Check and normalize
print(f"The original dataset shape is :{df.shape}")
from sklearn.preprocessing import minmax_scale
scaled_df = minmax_scale(df)
print(f"The scaled dataset shape is : {scaled_df.shape}")

# check column name
column_names = list(df.columns.values)
scaled_df = pd.DataFrame(scaled_df.tolist(),columns=column_names)

#Train_test_split
n = len(scaled_df)
train_df = scaled_df[0:int(n*0.7)]
val_df = scaled_df[int(n*0.7):int(n*0.9)]
test_df = scaled_df[int(n*0.9):]

num_features = scaled_df.shape[1]
print(f"The train shape is : {train_df.shape} \n"
      f"The valid shape is : {val_df.shape} \n"
      f"The test shape is : {test_df.shape} \n"
      f"The number of feature: {num_features}")

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     train_df=train_df,
                     val_df=val_df,
                     test_df = test_df,
                     label_columns=['T (degC)'])
print(w2)

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

print(example_window.shape)

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')