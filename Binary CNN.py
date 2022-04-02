import tensorflow as tf

print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import zipfile
import wget

# url = "http://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"
# filename = wget.download(url)
# print(filename)
# zip_ref = zipfile.ZipFile("pizza_steak.zip")
# zip_ref.extractall()
# zip_ref.close()
pizza_steak = "C:\\Users\\hustl\\PycharmProjects\\TensorFlowNew\\pizza_steak"
import os

for dirpath, dirnames, filenames in os.walk(pizza_steak):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

num_steak_images_train = len(os.listdir("C:\\Users\\hustl\\PycharmProjects\\TensorFlowNew\\pizza_steak\\train\\steak"))

print(num_steak_images_train)

import os

with os.scandir(pizza_steak) as entries:
    for entry in entries:
        print(entry.name)

with os.scandir("pizza_steak\\train") as entries:
    for entry in entries:
        print(entry.name)

with os.scandir("pizza_steak\\train\\steak") as entries:
    for entry in entries:
        print(entry.name)

import pathlib
import numpy as np
data_dir = pathlib.Path("pizza_steak/train/")
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)


# View an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");
  plt.show()

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img


# View a random image from the training dataset
img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="steak")

img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="pizza")

# View the img (actually just a big array/tensor)
print("Image Tensor:")
print(img)

# View the image shape
print("Image Shape:")
print(img.shape) # returns (width, height, colour channels)

# Get all the pixel values between 0 & 1
print("Normalized:")
print(img/255.)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the seed
tf.random.set_seed(42)

# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Setup the train and test directories
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42)

test_data = test_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)


# Get a sample of the training data batch
images, labels = train_data.next() # get the 'next' batch of images/labels
print("Number of Images and Labels:")
print(len(images)), print(len(labels))

# Create a CNN model (same as Tiny VGG - https://poloclub.github.io/cnn-explainer/)
model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10,
                         kernel_size=3, # can also be (3, 3)
                         activation="relu",
                         input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary activation output
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))


# Check out the layers in our model
model_1.summary()

# Plot the training curves
import pandas as pd
pd.DataFrame(history_1.history).plot(figsize=(10, 7));
plt.show()
# Plot the validation and training data separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()


  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


# Check out the loss curves of model_4
plot_loss_curves(history_1)
plt.show()

# View our example image
url = "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg"
filename = wget.download(url)
print(filename)
steak = mpimg.imread("03-steak.jpeg")

plt.imshow(steak)
plt.axis(False);
plt.show()

# Check the shape of our image
print(steak.shape)

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """
  # Read in target file (an image)
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

steak = load_and_prep_image("03-steak.jpeg")
print(steak)

# Add an extra axis
print(f"Shape before new dimension: {steak.shape}")
steak = tf.expand_dims(steak, axis=0) # add an extra dimension at axis 0
#steak = steak[tf.newaxis, ...] # alternative to the above, '...' is short for 'every other dimension'
print(f"Shape after new dimension: {steak.shape}")
print(steak)

# Make a prediction on custom image tensor
pred = model_1.predict(steak)
print(pred)

pred_class = class_names[int(tf.round(pred)[0][0])]
print(pred_class)

def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  pred_class = class_names[int(tf.round(pred)[0][0])]

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);

# Test our model on a custom image
pred_and_plot(model_1, "03-steak.jpeg", class_names)
plt.show()

pred_and_plot(model_1, "03-pizza-dad.jpeg", class_names)
plt.show()