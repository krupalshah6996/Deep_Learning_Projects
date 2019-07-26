#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:38:14 2019

@author: krupal
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing dataset

movies = pd.read_csv('ml-1m/movies.dat',header = None, encoding = 'Latin-1',engine = 'python', sep = '::')
ratings = pd.read_csv('ml-1m/ratings.dat',header = None, encoding = 'Latin-1',engine = 'python', sep = '::')
users = pd.read_csv('ml-1m/users.dat',header = None, encoding = 'Latin-1',engine = 'python', sep = '::')

#Preparing Test and Training set

training_set = pd.read_csv('ml-1m/training_set.csv')
test_set = pd.read_csv('ml-1m/test_set.csv')
training_set = np.array(training_set,dtype='int')
test_set = np.array(test_set,dtype='int')
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))
def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#Converting training and test sets in torch format for efficieny

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

class SAE(nn.Module):
    def __init__(self,):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies,40)
        self.fc2 = nn.Linear(40,20)
        self.fc3 = nn.Linear(20,40)
        self.fc4 = nn.Linear(40,nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

#Creating architecture of Autoencoder
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr = 0.10, weight_decay= 0.5)

#TRaining the model

nb_epochs = 300
for epoch in range(1,nb_epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output,target)
            mean_corrector = nb_movies/float((torch.sum(target.data > 0) + 1e-10))
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch:'+str(epoch)+' loss: '+str(train_loss/s))

#Testing the model
test_loss = 0
s = 0.
for id_user in range(0,nb_users):
     input = Variable(training_set[id_user]).unsqueeze(0)
     target = Variable(test_set[id_user]).unsqueeze(0)
     if torch.sum(target.data > 0) > 0:
         output = sae(input)
         target.require_grad = False
         output[target == 0] = 0
         loss = criterion(output,target)
         mean_corrector = nb_movies/float((torch.sum(target.data > 0) + 1e-10))
         test_loss += np.sqrt(loss.item()*mean_corrector)
         s += 1.
print(' loss: '+str(test_loss/s))
