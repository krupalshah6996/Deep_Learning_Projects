#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:49:04 2019

@author: krupal
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#PArt1: Input the dataset and load the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
 
#Scale the training_Set
 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

#Part2: Create a list which has 60 values since it is the timestamp for the RNN, 
# and it will be the set on which the RNN would be trained. Similarly, create the y_train 
#for training the RNN
 
X_train = []
y_train = []
for i in  range(120,1258):
    X_train.append(training_set_scaled[i-120:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train), np.array(y_train)

#Reshape the numpy array in the format specified in the Keras Documentation for RNN

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Part3: Creating the RNN with 4 LSTM layers

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(rate = 0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop',loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)

#Creating the test dataset for prediction of stock prices
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stocl_price = dataset_test.iloc[:,1:2].values

#Combination of both test and training datasets are needed for predicting the new stock prices
#Since each and every prediction requires 60 previous prediction, i.e. for ex: 20th Jan
#prediction requires stock prices of days in Jan as well

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-120:].values

#In order to allign the shape of the input with the required shape, rehsape function 
#with values -1,1 is used.

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#Predicting the stock prices

X_test=[]
for i in range(120,140):
    X_test.append(inputs[i-120:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the predicted and real stock price
plt.plot(real_stocl_price,color = 'red', label = 'real_stock_price')
plt.plot(predicted_stock_price,color = 'blue', label = 'predicted_stock_price')
plt.title('Google_Stock_Price')
plt.xlabel('Time')
plt.ylabel('Stock_price')
plt.legend()
plt.show()

#Using mean_squared_error for loss because we are predicting continuous values
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stocl_price, predicted_stock_price))
