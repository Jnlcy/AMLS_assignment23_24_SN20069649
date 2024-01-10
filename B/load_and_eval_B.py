import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
#
'''sys.path.append('./')
folder = 'B/'

model_filename = "pathmnist_model.h5"
data_flag = 'pathmnist'
'''


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
