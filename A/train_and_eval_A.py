import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib




def load_dataset(data_flag):

    input_root = 'Datasets/'
    
    dataset= np.load(os.path.join(input_root, "{}.npz".format(data_flag)))
    x_train = dataset['train_images']
    y_train = dataset['train_labels']
    x_test =  dataset['test_images']
    y_test = dataset['test_labels']
    x_val = dataset['val_images']
    y_val = dataset['val_labels']


    size = x_train[0].size
    X_train = x_train.reshape(x_train.shape[0], size, )
    X_val = x_val.reshape(x_val.shape[0], size, )
    X_test = x_test.reshape(x_test.shape[0], size, )

    return X_train,X_val,X_test,y_train,y_val,y_test


def data_scaling(X_train,X_val,X_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    return X_train,X_val,X_test



def model_selection(X_train,X_val,y_train,y_val, evaluation_metric='accuracy'):
# Define a list of classifiers to try
    classifiers = [
        ('Support Vector Machine', SVC(kernel='linear', C=1.0)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5))
    ]
    
    best_model = None
    best_model_name = None
    best_metric_value = -1  # Initialize with a low value for accuracy

    report_df = pd.DataFrame(columns=['Classifier', 'Classification Report'])
    # Iterate through classifiers and evaluate their performance
    for classifier_name, classifier in classifiers:
        print(f"Evaluating {classifier_name}...")
        
        # Train the classifier
        classifier.fit(X_train, y_train.ravel())
        
        # Predict on the validation set
        y_pred = classifier.predict(X_val)
        
        # Calculate the chosen evaluation metric on the validation set
        if evaluation_metric == 'accuracy':
            metric_value = accuracy_score(y_val, y_pred)

        # Calculate accuracy on the validation set
        print(f"{evaluation_metric.capitalize()} on validation set: {metric_value * 100:.2f}%")
        
        # Print classification report
        report = classification_report(y_val, y_pred, target_names=['Normal', 'Pneumonia'])

        # Append the classification report to the DataFrame
        
        print(report)
        # Check if this model is the best based on the chosen metric
        if metric_value > best_metric_value:
            best_model = classifier
            best_model_name = classifier_name
            best_metric_value = metric_value

        print("-" * 40)

    
    print(f'The best classifier on validation set is {best_model_name}')
    return best_model_name,best_model
#model_selection(X_train,X_val,y_train,y_val)
# confusion matrix
def confusion_matrix_plot(y_test,y_pred):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    return
def save_trained_model(best_classifier,model_filename):
    
    # Save the trained model to a file
    joblib.dump(best_classifier, model_filename)

    print(f"Trained model saved as {model_filename}")
    return
#load pretrained model and predict
def load_trained_model(folder,model_filename,data_flag):
    
    model_path = os.path.join(folder,model_filename)
    data_flag = 'pneumoniamnist'
    X_train,X_val,X_test,y_train,y_val,y_test =load_dataset(data_flag)
    X_train,X_val,X_test = data_scaling(X_train,X_val,X_test)
    if os.path.exists(model_path):

        # Load the saved model
        loaded_model = joblib.load(model_path)

        # Use the loaded model for predictions
        y_pred = loaded_model.predict(X_test)
        # Calculate accuracy on the test set
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on the test set : {accuracy * 100:.2f}%")
        confusion_matrix_plot(y_test,y_pred)
    else:
        print("Trained model does not exist, train the model first")
    return
#train model save model and predict
def train_and_save_classify():

    data_flag = 'pneumoniamnist'
    X_train,X_val,X_test,y_train,y_val,y_test =load_dataset(data_flag)
    X_train,X_val,X_test = data_scaling(X_train,X_val,X_test)

    best_classifier_name,best_classifier = model_selection(X_train,X_val,y_train,y_val)

    # Train the best classifier on the full training set
    best_classifier.fit(X_train, y_train.ravel())

    #save the trained classifier
    folder = "./A"
    # Define the filename for the saved model
    model_filename = os.path.join(folder,'pneumoniamnist_model.pkl')

    save_trained_model(best_classifier,model_filename)
    # Evaluate the best classifier on the test set
    y_pred = best_classifier.predict(X_test)

    # Calculate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set using {best_classifier_name}: {accuracy * 100:.2f}%")
    confusion_matrix_plot(y_test,y_pred)
    return


#train_save_pneu()
#load_trained_model()

'''param_grid = {
    'n_estimators': [300,400,600,1000],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2]  # Minimum number of samples required to be at a leaf node
}

# Create the Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=1, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train.ravel())

# Print the best hyperparameters found by grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best classifier with the optimal hyperparameters
best_classifier = grid_search.best_estimator_

# Predict on the test set using the best classifier
y_pred = best_classifier.predict(X_test)
'''
