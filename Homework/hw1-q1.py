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
        self.nClasses = n_classes

        mu, sigma = 0.1, 0.1

        # Initialize weight matrices with normal distribution N(mu, sigma^2)
        W1 = np.random.normal(mu, sigma, size = (hidden_size, n_features))
        W2 = np.random.normal(mu, sigma, size = (n_classes, hidden_size))
        
        # Initialize bias to zeroes vector
        b1 = np.zeros(hidden_size) # (hidden_size)
        b2 = np.zeros(n_classes) # (n_classes)

        self.weights = [W1, W2]
        self.nLayers = len(self.weights)
        self.biases = [b1, b2]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        predictedLabels = []

        for x in X:
            output, _ = self.forward(x)
            yHat = self.predictLabel(output)
            predictedLabels.append(yHat)

        return predictedLabels

    def evaluate(self, X, y):
        predictedLabels = self.predict(X)
        acc = np.mean(np.argmax(predictedLabels, axis = 1) == np.argmax(y, axis = 1))

        return acc

    def train_epoch(self, X, y, learning_rate=0.001):
        totalLoss = 0

        for x_i, y_i in zip(X, y):
            totalLoss += self.update_weights(x_i, y_i, learning_rate)

        print("Total loss: %f" % totalLoss)
    
    def update_weights(self, x, y, eta):
        z, hiddens = self.forward(x)

        # Compute loss
        probs = np.exp(z) / np.sum((np.exp(z))) # softmax
        loss = np.dot(-y, np.log(probs))

        gradWeights, gradBiases = self.backward(x, y, z, hiddens)

        # Update the weights and the biases
        for i in range(self.nLayers):
            self.weights[i] -= eta * gradWeights[i]
            self.biases[i] -= eta * gradBiases[i]

        return loss

    def forward(self, x):
        hiddenLayers = []

        for i in range(self.nLayers):
            h = x if i == 0 else hiddenLayers[i - 1]
            z = np.dot(self.weights[i], h) + self.biases[i]

            # If it isn't the last layer -> activation
            if(i < self.nLayers - 1):
                z = np.maximum(z, 0) # relu activation
                hiddenLayers.append(z)

        output = z

        return output, hiddenLayers

    def backward(self, x, y, output, hiddens):
        z = output

        # Cross-entropy loss function
        z -= np.max(z) # anti-overflow
        probs = np.exp(z) / np.sum((np.exp(z)))
        gradZ = probs - self.getOneHot(y)

        gradWeights = []
        gradBiases = []

        # for(i = nLayers - 1, i > -1, i--)
        # Basically a backwards "for" to access the hidden layers in the correct order
        for i in range(self.nLayers -1, -1, -1):
            h = x if i == 0 else hiddens[i -1]

            # Gradient of the current layer
            gradWeights.append(np.dot((gradZ[:, None]), h[:, None].T))
            gradBiases.append(gradZ)

            # Gradient of the previous layer
            gradH = np.dot(self.weights[i].T, gradZ)

            # Relu derivative
            h[h <= 0] = 0
            h[h > 0] = 1

            gradZ = gradH * h

        gradWeights.reverse()
        gradBiases.reverse()

        return gradWeights, gradBiases

    def predictLabel(self, output):
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1

        return y_hat
    
    def getOneHot(self, y):
        oneHot = np.zeros(self.nClasses)

        for i in range(self.nClasses):
            if i == y:
                oneHot[i] = 1
        
        return oneHot

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
