import sys, os
sys.path.append(os.curdir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

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

def relu(x):
    #y = (x > 0)*x
    return np.maximum(0,x)

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #preventing overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open('_ch03/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    #print(a1.shape)
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    #print(a2.shape)
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    #print(a3.shape)
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

"""
    My implementation 

total_cnt = 0
correct_cnt = 0
for xi, ti in zip(x, t):
    yi = np.argmax(predict(network, xi)) 
    total_cnt += 1
    if yi == ti:
        correct_cnt += 1
accuracy = correct_cnt/total_cnt
print(accuracy) #92.07%
"""

"""
    Textbook implementation
"""
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy: "+str(accuracy_cnt/len(x)))

    


