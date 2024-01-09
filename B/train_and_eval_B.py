import sys
import os

sys.path.append('./')



folder = 'B/'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_filename = "pathmnist_model.h5"
data_flag = 'pathmnist'

def get_data(data_flag):
    input_root = 'Datasets/'
    
    dataset= np.load(os.path.join(input_root, "{}.npz".format(data_flag)))
    x_train = dataset['train_images']
    y_train = dataset['train_labels']
    x_test =  dataset['test_images']
    y_test = dataset['test_labels']
    x_val = dataset['val_images']
    y_val = dataset['val_labels']

    X_train = x_train/255.0
    X_test = x_test/255.0
    X_val = x_val/255.0
    


    return X_train, y_train, X_test, y_test,X_val,y_val

# Load and preprocess your data
training_images, training_labels, test_images, test_labels,val_images, val_labels = get_data(data_flag)

# Convert labels to one-hot encoding
training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=9)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=9)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=9)

# Define the MLP model using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)

checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoint.h5', save_best_only=True, monitor='val_loss', mode='min')

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Fit the generator to your data
datagen.fit(training_images)

# Train the model with data augmentation
model.fit(datagen.flow(training_images, training_labels, batch_size=32),
          epochs=100,
          validation_data=(val_images, val_labels),
          callbacks=[early_stopping, checkpoint],
          verbose=1)  # Verbose for detailed output

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save(os.path.join(folder,model_filename))
