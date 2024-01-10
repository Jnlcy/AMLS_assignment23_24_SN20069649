import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')



folder = 'B/'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

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
    #normalizing
    X_train = x_train/255.0
    X_test = x_test/255.0
    X_val = x_val/255.0
    
    print(X_train[0].shape)

    return X_train, y_train, X_test, y_test,X_val,y_val

# Load and preprocess your data
training_images, training_labels, test_images, test_labels,val_images, val_labels = get_data(data_flag)
shuffle_indices = np.random.permutation(len(preprocessed_images))
# Convert labels to one-hot encoding
training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=9)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=9)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=9)

# Define the MLP model using Keras Sequential API
model = tf.keras.Sequential(name = 'Firstmodel')

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu' ))

model.add(layers.Dropout(0.25, name='Dropout3'))
model.add(layers.Dense(9, activation='softmax'))
model.summary()

#learning rate decay

#optimizer
opt = tf.keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=3,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')
    # Fit the generator to your data
datagen.fit(training_images)



epochs = 30
# Train the model
history = model.fit(datagen.flow(training_images, training_labels, batch_size=32),
          epochs=epochs,
          validation_data=(val_images, val_labels),
          callbacks=[early_stopping])
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save(os.path.join(folder,model_filename))

#plot history
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
title_fontsize = 16
axis_fontsize = 12

ax1.plot(range(1,len(history.history['loss'])+1), history.history['loss'], label='Training loss')
ax1.plot(range(1,len(history.history['loss'])+1), history.history['val_loss'], label='Validation Loss')
ax1.legend()
ax1.set_xticks(range(1,len(history.history['loss'])+1,3))
ax1.set_title('Loss', fontsize=title_fontsize)
ax1.set_xlabel('Epoch', fontsize=axis_fontsize)

ax2.plot(range(1,len(history.history['loss'])+1), history.history['accuracy'], label='Training Accuracy')
ax2.plot(range(1,len(history.history['loss'])+1), history.history['val_accuracy'], label='Validation Accuracy')
ax2.legend()
ax2.set_xticks(range(1,len(history.history['loss'])+1,3))
ax2.set_title('Accuracy', fontsize=title_fontsize)
ax2.set_xlabel('Epoch', fontsize=axis_fontsize) 
plt.show()

