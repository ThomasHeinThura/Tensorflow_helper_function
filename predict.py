
#Time series
def make_preds(model, input_data):
  import tensorflow as tf
  """
  Uses model to make predictions on input_data.

  Parameters
  ----------
  model: trained model 
  input_data: windowed input data (same kind of data model was trained on)

  Returns model predictions on input_data.

  Example: 
    model_1_preds = make_preds(model_1, test_windows)
    len(model_1_preds), model_1_preds[:10]

  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions


""" Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  
#   Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
#   Parameters
#   ---------
#   timesteps : array of timesteps
#   values : array of values across time
#   format : style of plot, default "."
#   start : where to start the plot (setting a value will index from start of timesteps & values)
#   end : where to end the plot (setting a value will index from end of timesteps & values)
#   label : label to show on plot of values

  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)

# Plot the prediction
def plot_prediciton():
    #Plot the prediction
    offset = 300
    plt.figure(figsize=(10, 7))
    # Account for the test_window offset and index into test_labels to ensure correct plotting
    plot_time_series(timesteps=X_test[-len(test_windows):], values=test_labels[:, 0], start=offset, label="Test_data")
    plot_time_series(timesteps=X_test[-len(test_windows):], values=model_1_preds, start=offset, format="-", label="model_1_preds")

    #Forest multi horizon (windows 30, hroizon 7)
    offset = 300
    plt.figure(figsize=(10, 7))
    plot_time_series(timesteps=X_test[-len(test_windows):], values=test_labels[:, 0], start=offset, label="Test_data")
    # Checking the shape of model_3_preds results in [n_test_samples, HORIZON] (this will screw up the plot)
    plot_time_series(timesteps=X_test[-len(test_windows):], values=model_3_preds, start=offset, label="model_3_preds")

    offset = 300
    plt.figure(figsize=(10, 7))
    # Plot model_3_preds by aggregating them (note: this condenses information so the preds will look fruther ahead than the test data)
    plot_time_series(timesteps=X_test[-len(test_windows):], 
                    values=test_labels[:, 0], 
                    start=offset, 
                    label="Test_data")
    plot_time_series(timesteps=X_test[-len(test_windows):], 
                    values=tf.reduce_mean(model_3_preds, axis=1), 
                    format="-",
                    start=offset, 
                    label="model_3_preds")

"""


### NLP
def predict_on_sentence(model, sentence):
    import tensorflow as tf
    """
    Uses model to make a prediction on sentence.

    Returns the sentence, the predicted label and the prediction probability.
    """
    pred_prob = model.predict([sentence])
    pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
    print(f"Pred: {pred_label}", "(real disaster)" if pred_label > 0 else "(not real disaster)", f"Prob: {pred_prob[0][0]}")
    print(f"Text:\n{sentence}")

  
# Calculate the time of predictions
def show_time_taken_to_predict(model, samples):
    import time
    """
    Times how long a model takes to make predictions on samples.
  
    Args:
    ----
    model = a trained model
    sample = a list of samples

    Returns:
    ----
    total_time = total elapsed time for model to make predictions on samples
    time_per_pred = time in seconds per single sample
    """
    start_time = time.perf_counter() # get start time
    model.predict(samples) # make predictions
    end_time = time.perf_counter() # get finish time
    total_time = end_time-start_time # calculate how long predictions took to make
    time_per_pred = total_time/len(samples) # find prediction time per sample
    return total_time, time_per_pred

""" Time and preformace trade off Procedures
# Calculate TF Hub Sentence Encoder prediction times
model_6_total_pred_time, model_6_time_per_pred = pred_timer(model_6, val_sentences)
model_6_total_pred_time, model_6_time_per_pred

# Calculate Naive Bayes prediction times
baseline_total_pred_time, baseline_time_per_pred = pred_timer(model_0, val_sentences)
baseline_total_pred_time, baseline_time_per_pred

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.scatter(baseline_time_per_pred, baseline_results["f1"], label="baseline")
plt.scatter(model_6_time_per_pred, model_6_results["f1"], label="tf_hub_sentence_encoder")
plt.legend()
plt.title("F1-score versus time per prediction")
plt.xlabel("Time per prediction")
plt.ylabel("F1-Score");
"""


### Computer vision
# prepare to predict certain image
def reshape_image_to_predict(filename, img_shape=224, scale=True):
    import tensorflow as tf
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
    import tensorflow as tf
    import matplotlib.pyplot as plt
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = reshape_image_to_predict(filename)

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

def predict_image_from_dir(model, class_names, test_dir):
    # Make preds on a series of random images
    import os
    import random
    import matplotlib.pyplot as plt

    plt.figure(figsize=(17, 10))
    for i in range(3):
        # Choose a random image from a random class
        #class_names = test_data.class_names
        class_name = random.choice(class_names)
        filename = random.choice(os.listdir(test_dir + "/" + class_name))
        filepath = test_dir + class_name + "/" + filename

        # Load the image and make predictions
        img = reshape_image_to_predict(filepath, scale=False)  # don't scale images for EfficientNet predictions
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
    import random
    import tensorflow as tf
    import matplotlib.pyplot as plt
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
