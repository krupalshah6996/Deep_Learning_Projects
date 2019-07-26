#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:56:53 2019

@author: krupal
"""
#importing libraries and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#PArt1: Input the dataset and load the training set

dataset_train = pd.read_csv('Credit_Card_Applications.csv')
X = dataset_train.iloc[:,:-1].values
y = dataset_train.iloc[:,-1].values

#Scaling the data

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Creating SOM

from minisom import MiniSom
som = MiniSom(x=10, y=10, learning_rate = 0.5, sigma = 1.0, input_len = 15)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualization

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()

#Detecting frauds

markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor = colors[y[i]],markerfacecolor = 'None',markersize = 10,markeredgewidth = 2)
show()
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,3)],mappings[(8,5)]), axis = 0)
frauds = sc.inverse_transform(frauds)
