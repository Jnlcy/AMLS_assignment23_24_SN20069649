import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')



folder = 'B/'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras.models import Sequential

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

training_images=tf.image.resize(training_images,(64,64))
test_images =tf.image.resize(test_images,(64,64))
val_images=tf.image.resize(test_images,(64,64))

# Convert labels to one-hot encoding
training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=9)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=9)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=9)







# Define the MLP model using Keras Sequential API
model=Sequential([
    layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(64,64,3)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3)),
    layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1024,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(9,activation='softmax')  
    
    
])


model.summary()
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate= 0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')
    # Fit the generator to your data
datagen.fit(training_images)



epochs = 50
# Train the model
history = model.fit(datagen.flow(training_images, training_labels, batch_size=32),
          epochs=epochs,
          validation_data=(val_images, val_labels),
          callbacks=[early_stopping])
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


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

# Save the model
model.save(os.path.join(folder,model_filename))
