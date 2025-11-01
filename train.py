from sklearn.datasets import fetch_openml
from model import backprop, one_hot_encode, forwardprop, softmax, compute_loss, relu_derivative, relu, initialize_parameters
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)
X_train = X_train.T
X_test = X_test.T

Y_train = one_hot_encode(y_train)
Y_test = one_hot_encode(y_test)

epochs = 500
parameters = initialize_parameters(784, 10, 10)

for i in range(epochs):
    cache = forwardprop(X_train, parameters)
    A2 = cache["a2"]
    loss = compute_loss(A2, Y_train)
    grads = backprop(X_train, Y_train, parameters, cache)
    
    lr = 0.7

    parameters["w1"] -= lr * grads["dw1"]
    parameters["b1"] -= lr * grads["db1"]
    parameters["w2"] -= lr * grads["dw2"]
    parameters["b2"] -= lr * grads["db2"]

    if i%50==0:
        print(f"Epoch {i}: loss = {loss:.4f}")