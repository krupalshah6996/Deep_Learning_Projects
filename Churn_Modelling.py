# Part 1 - Data Preprocessing

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib as plt

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_X1 = LabelEncoder()
label_X2 = LabelEncoder()
X[:,1] = label_X1.fit_transform(X[:,1])
X[:,2] = label_X1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - The ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation = 'relu',kernel_initializer='uniform',input_dim = 11))
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(6, activation = 'relu',kernel_initializer='uniform'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid',kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train,y_train,batch_size=5,epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation = 'relu',kernel_initializer='uniform',input_dim = 11))
    classifier.add(Dense(6, activation = 'relu',kernel_initializer='uniform'))
    classifier.add(Dense(1, activation = 'sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier                                                                                                 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator= classifier,X = X_train, y = y_train, cv = 10, n_jobs=-1)
accu_mean = accuracies.mean()
acc_std = accuracies.std()

# Improving the ANN

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
