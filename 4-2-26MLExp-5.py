#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Feature (X) and Target (y)

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()












# In[12]:


#Implement KNN Algorithm 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=4)

# Plot original data
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='.', s=100, edgecolors='black')
plt.title("Original Data")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train KNN models
knn5 = KNeighborsClassifier(5).fit(X_train, y_train)
knn1 = KNeighborsClassifier(1).fit(X_train, y_train)

# Predict
y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)

# Plot predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_5, marker='*', s=100, edgecolors='black')
plt.title("k = 5")

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_1, marker='*', s=100, edgecolors='black')
plt.title("k = 1")

plt.show()


# In[ ]:




