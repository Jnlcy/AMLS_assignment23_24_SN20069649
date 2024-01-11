import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')



import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras.models import Sequential
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model



#preprocess
def load_and_preprocess_data(data_flag, input_root='Datasets/'):
    dataset = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))
    x_train, y_train = dataset['train_images'] / 255.0, dataset['train_labels']
    x_test, y_test = dataset['test_images'] / 255.0, dataset['test_labels']
    x_val, y_val = dataset['val_images'] / 255.0, dataset['val_labels']

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=9)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=9)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=9)

    return x_train, y_train, x_test, y_test, x_val, y_val


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

#model definition
def create_model():
    model = tf.keras.Sequential(name='Firstmodel')
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25, name='Dropout3'))
    model.add(layers.Dense(9, activation='softmax'))
    
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#training function
def train_model(model, training_images, training_labels, val_images, val_labels, epochs=50):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True)

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2,
                                 zoom_range=0.1, horizontal_flip=False,
                                 fill_mode='nearest')
    datagen.fit(training_images)

    history = model.fit(datagen.flow(training_images, training_labels, batch_size=16),
                        epochs=epochs, validation_data=(val_images, val_labels),
                        callbacks=[early_stopping])
    return history

#evaluating function
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_loss, test_acc

# plotting function
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Training loss')
    ax1.plot(range(1, len(history.history['loss']) + 1), history.history['val_loss'], label='Validation Loss')
    ax1.legend()
    ax1.set_xticks(range(1, len(history.history['loss']) + 1, 3))
    ax1.set_title('Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)

    ax2.plot(range(1, len(history.history['loss']) + 1), history.history['accuracy'], label='Training Accuracy')
    ax2.plot(range(1, len(history.history['loss']) + 1), history.history['val_accuracy'], label='Validation Accuracy')
    ax2.legend()
    ax2.set_xticks(range(1, len(history.history['loss']) + 1, 3))
    ax2.set_title('Accuracy', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    plt.show()
    return

def plot_confusion_matrix(model,X_test,y_test):
    to_Pred = X_test.reshape((7180, 28, 28,3))
    pred = model.predict(to_Pred)
    y_pred = pred.argmax(axis=1).tolist()

    rounded_labels=np.argmax(y_test, axis=1)

    class_names = ['adipose', 'background', 'debris','lymphocytes', 'mucus',  'smooth muscle',  'normal colon mucosa', 'cancer-associated stroma',  'colorectal adenocarcinoma epithelium']
    # Compute confusion matrix
    cm = confusion_matrix(rounded_labels, y_pred)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    return


#save the model
def save_model(model, folder, model_filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_path = os.path.join(folder, model_filename)
    model.save(model_path)

def load_CNN_model(data_flag):
    _,_, X_test, y_test,_,_ = get_data(data_flag)

    # Load the saved model
    model_path = 'B/pathmnist_model.h5'
    model = load_model(model_path)


    #Convert labels to one-hot encoding
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=9)  # If your labels are not already one-hot encoded

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}, Test loss: {test_loss}")
    plot_confusion_matrix(model,X_test,y_test)
    return

# Load and preprocess data
def train_and_eval_B():
    x_train, y_train, x_test, y_test, x_val, y_val = load_and_preprocess_data('pathmnist')

    # Create and train model
    model = create_model()
    history = train_model(model, x_train, y_train, x_val, y_val)

    # Evaluate and save model
    evaluate_model(model, x_test, y_test)
    plot_confusion_matrix(x_test, y_test)
    plot_training_history(history)
    save_model(model, 'B/', 'pathmnist_model.h5')
    return


'''
# Load and preprocess your data
training_images, training_labels, test_images, test_labels,val_images, val_labels = get_data(data_flag)



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


# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=3,         # Number of epochs with no improvement after which training will be stopped
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
history = model.fit(datagen.flow(training_images, training_labels, batch_size=16),
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
'''