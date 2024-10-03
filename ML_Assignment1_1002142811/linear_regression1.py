#!/usr/bin/env python
# coding: utf-8

# In[7]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        
        self.loss_history = []

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        lr = 0.01
        #learning rate
       
        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = None
        self.bias = None
        
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        ratio = 0.9 
        #train to validate ratio
        train_size = int(ratio*len(X))
        
        X_train , X_val = X[0:train_size] , X[train_size:]
        y_train , y_val = y[0:train_size] , y[train_size:]
        #splicing
        
        best_loss = float('inf')
        consecutive_increases = 0
        best_weights, best_bias = self.weights, self.bias
        
        # Gradient calculation function
        def calculate_gradients(X, y, weights, bias):
            m = len(X)
            predictions = np.dot(X, weights) + bias
            grad_weights = (1 / m) * np.dot(X.T, (predictions - y))
            grad_bias = (1 / m) * np.sum(predictions - y)
            return grad_weights, grad_bias
        
        for epoch in range(max_epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Calculate gradients and update parameters
                grad_weights, grad_bias = calculate_gradients(batch_X, batch_y, self.weights, self.bias)
                self.weights -= lr*grad_weights
                self.bias -= lr*grad_bias
                
            # Calculate the loss on the validation set
            val_predictions = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((val_predictions - y_val) ** 2)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights, best_bias = self.weights.copy(), self.bias
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases >= patience:
                    break
            self.loss_history.append(val_loss)
        
        # Set the model's parameters to the best values found during training
        self.weights, self.bias = best_weights, best_bias
        # TODO: Implement the training loop.

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        
        predictions = self.predict(X)
        m = y.shape[1]

        # return the Mean Squared Error (MSE)
        
        mse = m * np.mean((predictions - y) ** 2)

        return mse
    
    
    def save(self, file_path):
        # Save model parameters to a file
        np.savez(file_path, weights=self.weights, bias=self.bias)
    
    def load(self, file_path):
        # Load model parameters from a file
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']


# In[ ]:




