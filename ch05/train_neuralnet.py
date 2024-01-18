import sys,os
sys.path.append(os.curdir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

import matplotlib.pylab as plt


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

# Hyper Parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


for i in range(iters_num):
    #print('Iteration: '+str(i)+' in progress...')
    # Get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Compute gradients
    #grad = network.numerical_gradient(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    # Update parameters (W, b)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]
        
    # Record training progress
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #accuracy = network.accuracy(x_batch, t_batch)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("i " + str(i) + ": train acc, test acc | "+ str(train_acc) + ", " + str(test_acc))


#plt.plot(range(0,iters_num),train_loss_list)

plt.plot(range(0, int(iters_num//iter_per_epoch)+1), train_acc_list, test_acc_list)
plt.show()

