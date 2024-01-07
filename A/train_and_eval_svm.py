import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report



input_root = 'Datasets/'
data_flag = 'pneumoniamnist'
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


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
classifiers = [
        ('Support Vector Machine', SVC(kernel='linear', C=1.0)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5))
    ]
def model_selection(classifiers,X_train,X_val,y_train,y_val):
# Define a list of classifiers to try
    
    report_df = pd.DataFrame(columns=['Classifier', 'Classification Report'])
    # Iterate through classifiers and evaluate their performance
    for classifier_name, classifier in classifiers:
        print(f"Evaluating {classifier_name}...")
        
        # Train the classifier
        classifier.fit(X_train, y_train.ravel())
        
        # Predict on the validation set
        y_pred = classifier.predict(X_val)
        
        # Calculate accuracy on the validation set
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
        
        # Print classification report
        report = classification_report(y_val, y_pred, target_names=['Normal', 'Pneumonia'])
        # Append the classification report to the DataFrame
        report_df = report_df.append({'Classifier': classifier_name, 'Classification Report': report}, ignore_index=True)
        print(report)
        
        print("-" * 40)

    report_df.to_csv('classification_report.xlsx', index=False)
#model_selection(X_train,X_val,y_train,y_val)


# Select the best performing classifier based on validation results
best_classifier_name = 'Random Forest'  # Change this based on your evaluation
best_classifier = [clf for clf in classifiers if clf[0] == best_classifier_name][0]

# Train the best classifier on the full training set
best_classifier[1].fit(X_train, y_train.ravel())

# Evaluate the best classifier on the test set
y_pred = best_classifier[1].predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set using {best_classifier_name}: {accuracy * 100:.2f}%")

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
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

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")