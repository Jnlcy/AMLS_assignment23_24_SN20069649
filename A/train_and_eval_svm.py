import numpy as np
import os

import pandas as pd
from skimage.transform import resize 
from skimage.io import imread 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn import svm 


input_root = 'Datasets/'
data_flag = 'pneumoniamnist'
npz= np.load(os.path.join(input_root, "{}.npz".format(data_flag)))
x_train = npz['train_images']
y_train = npz['train_labels']
x_test =  npz['test_images']
y_test = npz['test_labels']
x_val = npz['val_images']
y_val = npz['val_labels']

size = x_train[0].size
X_train = x_train.reshape(x_train.shape[0], size, )
X_val = x_val.reshape(x_val.shape[0], size, )
X_test = x_test.reshape(x_test.shape[0], size, )



# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['rbf','poly']} 
  
# Creating a support vector classifier 
svc=svm.SVC(probability=True) 
  
# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param_grid)
model.fit(X_train,y_train)

   



    
    
