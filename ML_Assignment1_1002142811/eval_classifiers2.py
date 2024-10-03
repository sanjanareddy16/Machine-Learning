#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from LDA import LDA

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 1]]  # sepal Length and Width as input features
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Load the trained Logistic Regression model parameters for this variant
model1 = LogisticRegression()
model1.load("logistic_regression_params_1.npz")
model2 = LDA()
model2.load("lda_params_1.npz")

# Make predictions on the test data
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

# Calculate and print the accuracy
accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Accuracy for sepal Length/Width variant logistic regression: {accuracy1:.4f}")
print(f"Accuracy for sepal Length/Width variant LDA: {accuracy2:.4f}")


# In[ ]:




