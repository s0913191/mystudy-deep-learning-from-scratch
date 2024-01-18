import numpy as np

def step_function(x):
    y = x > 0 # true/false
    return y.astype(np.int64)

def sigmoid(x):
    """
        To avoid RuntimeWarning: overflow encountered in exp
        https://www.kamishima.net/mlmpyja/lr/sigmoid.html
    """
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)
    y = 1.0 / (1.0 + np.exp(-x))
    return y

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    #y = (x > 0)*x
    return np.maximum(0,x)

def identity_function(x):
    return x

"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #preventing overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
"""

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def sum_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + delta)) / batch_size

    

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()

    return grad



def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for step in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x
