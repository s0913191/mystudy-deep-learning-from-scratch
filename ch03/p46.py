import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0 # true/false
    return y.astype(np.int64)

def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y

def relu(x):
    #y = (x > 0)*x
    return np.maximum(0,x)

def identity_function(x):
    return x

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1, 0.4],[0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Z3 = identity_function(A3)


print(W1.shape)
print(X.shape)
print(B1.shape)
print(A1)
print(Z1)

print(A2)
print(Z2)

print(A3)
print(Z3)