#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LDA:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        self.weights = np.zeros((n_features, len(classes)))
        self.bias = np.zeros(len(classes))
        
        for c in classes:
            X_c = X[y == c]
            self.weights[:, c] = np.mean(X_c, axis=0)
            self.bias[c] = np.log(len(X_c) / n_samples)
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.argmax(linear_model, axis=1)
    
    def save(self, file_path):
        """Save model parameters (weights and bias) to a file using NumPy's np.savez."""
        np.savez(file_path, weights=self.weights, bias=self.bias)
    
    def load(self, file_path):
        """Load model parameters (weights and bias) from a file."""
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']



# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 11)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Variants of input features
variants = [
    {"features": [0, 1], "title": "Sepal Length/Width"},
    {"features": [2, 3], "title": "Petal Length/Width"},
    {"features": [0, 1, 2, 3], "title": "All Features"},
]

plot_variants = [
    {"features": [0, 1], "title": "Sepal Length/Width"},
    {"features": [2, 3], "title": "Petal Length/Width"},
]

# Create and train the LDA model
# Initialize and train the Logistic Regression model for each variant
models = []
i = 0
for variant in variants:
    i = i + 1
    features = variant["features"]
    X_train_variant = X_train[:, features]
    X_test_variant = X_test[:, features]
    model = LDA()
    model.fit(X_train_variant, y_train)
    model.save(f"lda_params_{i}.npz")
    models.append(model)
    
    # Visualize the decision boundary for the current variant
    if variant in plot_variants:
        plot_decision_regions(X_train_variant, y_train, clf=model, legend=2)
        plt.xlabel(iris.feature_names[features[0]])
        plt.ylabel(iris.feature_names[features[1]])
        plt.title(f"Linear Discriminant Analysis - {variant['title']}")
        plt.show()


# In[ ]:




