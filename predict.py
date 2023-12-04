import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping



RANDOM_SEED = 123



################ VISUALISE THE DATA BEFORE SPLITING ##########################
from data_visualization import visualize_image_distribution
yes_folder = 'data/yes'
no_folder = 'data/no'
visualize_image_distribution(yes_folder, no_folder)


################ SPLIT DATA ##########################

from split_data import split_dataset
original_dataset_path = 'data'
new_dataset_path = 'dataset'
split_dataset(original_dataset_path, new_dataset_path)


################ VISUALISE THE DATA AFTER SPLITING ##########################
from data_visualization import visualize_data
visualize_data()


################ PLOT SAMPLE IMAGES ##########################
from show_img import visualize_img
visualize_img()



################ NORMALIZE THE IMAGES ##########################
from normalize import resize_images
train_folder = 'dataset/train'
val_folder = 'dataset/validation'
test_folder = 'dataset/test'

normalized_train_folder = 'normalized_dataset/train'
normalized_val_folder = 'normalized_dataset/validation'
normalized_test_folder = 'normalized_dataset/test'

resize_images(train_folder, normalized_train_folder)
resize_images(val_folder, normalized_val_folder)
resize_images(test_folder, normalized_test_folder)


################ PERFORM DATA AUGMENTATION ##########################
"""from augment_data import augment_data
original_yes_folder = 'normalized_dataset/train/yes'
original_no_folder = 'normalized_dataset/train/no'
augmented_yes_folder = 'normalized_dataset/train_set/yes'
augmented_no_folder = 'normalized_dataset/train_set/no'
num_augmented_images = 6
augment_data(original_yes_folder, augmented_yes_folder, 'yes', num_augmented_images)
augment_data(original_no_folder, augmented_no_folder, 'no', num_augmented_images)"""









from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Constants
input_shape = (224, 224, 3)  # Assuming resized images are 224x224 with 3 channels (RGB)
num_classes = 1  # Binary classification (tumor or no tumor)

# Define the model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output before feeding into the dense layer
model.add(Flatten())

# Dense layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization

model.add(Dense(num_classes, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Training parameters
batch_size = 32
epochs = 30

# Use ImageDataGenerator for data augmentation during training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    normalized_train_folder,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    normalized_val_folder,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    verbose=1
)


# Plot training and validation accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Evaluate the model on the validation set
import seaborn as sns
import math
val_generator.reset()

# Calculate the number of steps needed for the generator
val_steps = math.ceil(val_generator.samples / val_generator.batch_size)

# Predict and evaluate
predictions = model.predict_generator(val_generator, steps=val_steps, verbose=1)
predicted_classes = (predictions > 0.8).astype('int')  # Assuming threshold for binary classification is 0.5

true_classes = val_generator.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




# TEST THE MODEL
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Constants
input_shape = (224, 224, 3)  # Assuming resized images are 224x224 with 3 channels (RGB)
batch_size = 32
# Use ImageDataGenerator for data normalization during testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    normalized_test_folder,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate_generator(test_generator, steps=None, verbose=True)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')




## Test on a specific image
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to the image you want to test
test_image_path = 'data/yes/Y59.jpg'

# Load and preprocess the image
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values to be in the range [0, 1]

# Make predictions
prediction = model.predict(img_array)

# Assuming binary classification with a threshold of 0.5
class_label = 'Tumor' if prediction > 0.8 else 'No Tumor'

# Display the image
plt.imshow(img)
plt.title(f'Predicted Class: {class_label}')
plt.axis('off')
plt.show()





