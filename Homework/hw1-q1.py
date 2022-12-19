#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    """
    z - (number of classes x number of examples)
    """
    z -= np.max(z) # avoids overflow
    Z = (np.exp(z)).sum(axis=0) # (1 x n_examples)
    output_probabilities = np.exp(z) / Z # (n_classes x n_examples)
    return output_probabilities


def reluDerivative(z): 
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def forward(x, w, b):
    """
    Implements the linear pre-activation of a layer
    W - weights matrix - (size of current layer x size of previous layer)
    h - results from previous layer (or input data) - (number of examples x size of previous layer)
    b - bias vector - (size of the current layer)
    """
    nLayers = len(w)
    hiddens = []
    x = x.reshape(-1, 1)

    for i in range(nLayers):
        h = x if i == 0 else hiddens[i -1]
        print(h.shape)
        print(w[i].shape)
        print(b[i].shape)
        z = w[i].dot(h) + b[i]
        #import ipdb; ipdb.set_trace()
        if(i < nLayers - 1): 
            hiddens.append(relu(z)) # activate the layers except the output layers
    print(z.shape)
    print(x.shape)

    return z, hiddens

def backward(x, y, z, hiddens, W, B):
    nLayers = len(W)
    
    probs = softmax(z)
    gradZ = probs - y
    #print(probs.shape)
    #print(y.shape)
    #print(gradZ.shape)
    
    gradW = []
    gradB = []

    for i in range(nLayers -1):
        h = x if i == 0 else hiddens[i -1]
        print(i)
        gradW.append(np.dot(gradZ[:, None], h[:, None].T))
        gradB.append(gradZ)

        gradH = np.dot(W[i].T, gradZ)

        gradZ = gradH * reluDerivative(gradZ)

    gradW.reverse()
    gradB.reverse()

    return gradW, gradB

def cross_entropy_loss(y_probabilities, y):
    """
    y_probabilities - probability vector from our prediction (n_examples)
    y - gold label, one-hot vector (n_examples)
    """
    print(y)
    print(y_probabilities)
    #import ipdb; ipdb.set_trace()

    one_hot_index = y[0]
    y = np.zeros((10,1))
    y[one_hot_index] = 1

    #import ipdb; ipdb.set_trace()

    loss = np.dot(-y.T, np.log(y_probabilities))
    return loss

def predict_label(z):
    y_hat = np.zeros_like(z)
    y_hat[np.argmax(z, axis=0)] = 1
    return y_hat

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features)) #Does our weight already have bias? np.zeros((n_classes, n_features + 1 ))....

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a - I think this is correct

        # Calculate highest score -> get predicted class
        y_hat = np.dot(self.W, x_i).argmax(axis=0)

        #If a mistake is committed, correct it
        if(y_hat != y_i):
            self.W[y_i, :] += x_i.T # Increase weight of gold class
            self.W[y_hat, :] -= x_i.T # Decrease weight of incorrect class

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b - I think this is correct

        # Calculate scores according to the model (n_classes x 1).
        scores = np.dot(self.W, x_i)[:,None]

        # One-hot vector with the gold label (n_classes x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Conditional probability of y, according to softmax (n_classes x 1).
        # y_probabilities = Softmax(scores) not working properly
        z = np.sum(np.exp(scores))
        y_probabilities = np.exp(scores) / z

        # Update weights with stochastic gradient descent
        self.W += learning_rate * (y_one_hot - y_probabilities) * x_i[None, :]
        


class MLP(object):
    # Q2.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize weight matrices with normal distribution N(mu, sigma^2)
        mu, sigma = 0.1, 0.1

        W1 = np.random.normal(mu, sigma, hidden_size * n_features)
        W1 = np.reshape(W1, (hidden_size, n_features)) # (hidden_size x n_features)

        W2 = np.random.normal(mu, sigma, n_classes * hidden_size)
        W2 = np.reshape(W2, (n_classes, hidden_size)) # (n_classes x hidden_size)
        
        # Initialize bias to zeroes vector
        b1 = np.zeros((hidden_size, 1)) # (hidden_size)
        b2 = np.zeros((n_classes, 1)) # (n_classes)

        self.weights = [W1, W2]
        self.biases = [b1, b2]

    def predict(self, X):
        """
        X (n_examples x n_features):
        """
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        predicted_labels = []

        for x in X:
            output, _ = forward(x, self.weights, self.biases)
            y_hat = predict_label(output)
            predicted_labels.append(y_hat)

        predicted_labels = np.array(predicted_labels)

        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        predicted_labels = self.predict(X)
        acc = np.mean(np.argmax(predicted_labels, axis = 0) == np.argmax(y, axis = 0))

        return acc

    def update_weights(self, gradW, gradB, learning_rate):
        nLayers = len(self.weights)
        print(nLayers)
        print(len(gradB))
        print(len(gradW))

        for i in range(nLayers):
            self.weights[i] -= learning_rate * gradW[i]
            self.biases[i] -= learning_rate * gradB[i]


    def train_epoch(self, X, y, learning_rate=0.001):
        totalLoss = 0

        print("X shape")
        print(X.shape)

        y = y.reshape(-1, 1)
        print("y shape")
        print(y.shape)


        for x, yy in zip(X, y):
            x = x.reshape(-1, 1)
            yy = yy.reshape(-1, 1)

            output, hiddens = forward(x, self.weights, self.biases)
            print("ys")
            print(x.shape)
            print(yy.shape) # SOMETIMES YY IS ()
            loss = cross_entropy_loss(softmax(output), yy)
            totalLoss += loss
            gradWeights, gradBiases = backward(x, yy, output, hiddens, self.weights, self.biases)
            self.update_weights(gradWeights, gradBiases, learning_rate)
        
        return loss


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
