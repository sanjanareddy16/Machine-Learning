#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from linearregressionwithl2 import LinearRegressionwithL2
from linear_regression1 import LinearRegression


# In[6]:


from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# The dataset is stored in the 'data' and 'target' attributes    
X = iris.data[:, [2, 1]]  # Using Petal Length and Sepal Width as input features
y = iris.data[:, 3]       # Using Petal Width as the output feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)

modell2 = LinearRegressionwithL2(batch_size=32, regularization=0.1, max_epochs=100, patience=3)
modell2.load('model_regression1withl2.npz') 

model = LinearRegression(batch_size=32, max_epochs=100, patience=3)
model.load('model_regression1.npz') 

# Calculate the absolute differences between parameters
weight_differences = np.abs(modell2.weights - model.weights)
bias_difference = np.abs(modell2.bias - model.bias)

# Record the differences
parameter_differences = {
    "Weight Differences": weight_differences,
    "Bias Difference": bias_difference,
}

# Print or access the recorded differences
for param, diff in parameter_differences.items():
    print(param, ":", diff)
    


# In[ ]:




