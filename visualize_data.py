# 2.1 visualize plot data
# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  import matplotlib.pyplot as plt
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values

  Example useage:
  plt.figure(figsize=(10, 7))
  plot_time_series(timesteps=X_train, values=y_train, label="Train data")
  plot_time_series(timesteps=X_test, values=y_test, label="Test data")

  """
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("Features")
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)


# 2.3 visualize picture
# Plot multiple random images of fashion MNIST
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

# 2.4 visualize words and pandas
def visualize_words(dataset):
  # Let's visualize some random training examples
  import random
  random_index = random.randint(0, len(dataset)-5) # create random indexes not higher than the total number of samples
  for row in dataset[["text", "target"]][random_index:random_index+5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
    print(f"Text:\n{text}\n")
    print("---\n")


