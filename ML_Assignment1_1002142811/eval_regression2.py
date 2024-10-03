#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from linear_regression1 import LinearRegression


# In[6]:


from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
 
X = iris.data[:, [3, 0]]  # Using Petal Width and Sepal Length as input features
y = iris.data[:, 2]       # Using Petal Length as the output feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)

model = LinearRegression(batch_size=32, max_epochs=100, patience=3)
model.load('model_regression2.npz') 

# Evaluate the model on the test dataset
mse = model.score(X_test, y_test)

# Print the MSE 
print(f"Mean Squared Error on Test Data (Model 2): {mse:.4f}") 


# In[ ]:





# In[ ]:




