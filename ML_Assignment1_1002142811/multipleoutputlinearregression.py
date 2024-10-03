#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class LinearRegressionMultipleOutputs:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """
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

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """

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

        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = None
        self.bias = None
        
        n_features = X.shape[1]
        m_outputs = y.shape[1]  # Number of output dimensions
        self.weights = np.zeros((n_features, m_outputs))
        self.bias = np.zeros(m_outputs)
        
        ratio = 0.9  # Train to validate ratio
        train_size = int(ratio * len(X))
        
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        best_loss = float('inf')
        consecutive_increases = 0
        best_weights, best_bias = self.weights, self.bias
        
        for epoch in range(max_epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Calculate gradients and update parameters
                predictions = np.dot(batch_X, self.weights) + self.bias
                error = predictions - batch_y
                grad_weights = (1 / len(batch_X)) * np.dot(batch_X.T, error)
                grad_bias = (1 / len(batch_X)) * np.sum(error, axis=0)

                self.weights -= lr * grad_weights
                self.bias -= lr * grad_bias
                
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
        
        # Set the model's parameters to the best values found during training
        self.weights, self.bias = best_weights, best_bias

    def predict(self, X):
        """

        Parameters:
        X: numpy.ndarray
            The input data.
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """
        Parameters:
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        predictions = self.predict(X)
        n = X.shape[0]
        m = y.shape[1]  # Number of output dimensions

        # Calculate the Mean Squared Error (MSE) 
        mse = (1/(n*m))*np.sum((predictions - y) ** 2)
        return mse

    def save(self, file_path):
        # Save model parameters to a file
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load(self, file_path):
        # Load model parameters from a file
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']

        


# Load the Iris dataset
iris = load_iris()


X = iris.data[:, [0, 1]]  # Using Sepal Length and Sepal Width as input features

y = iris.data[:, [2, 3]]  # Using Petal Length and Petal Width as output features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)




# Initialize the model
model = LinearRegressionMultipleOutputs()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) for multiple outputs

mse = model.score(X_test, y_test)
print(f"Mean Squared Error (MSE) for Multiple Outputs: {mse:.4f}")

# Plot the actual vs. predicted values for both petal length and width
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0])
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Petal Length - Actual vs. Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1])
plt.xlabel("Actual Petal Width")
plt.ylabel("Predicted Petal Width")
plt.title("Petal Width - Actual vs. Predicted")

plt.tight_layout()
plt.show()


# In[ ]:




