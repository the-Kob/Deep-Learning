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

        # Calculate predicted class (Lecture 3, 17)
        #y_hat = np.dot(self.W, x_i).argmax(axis=0)

        # Stochastic gradient descent formula
        #self.W += learning_rate * (y_i - y_hat) * x_i.T

        # Calculate scores according to the model (n_classes x 1).
        scores = np.dot(self.W, x_i)[:,None]

        # One-hot vector with the gold label (n_classes x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Conditional probability of y, according to softmax (n_classes x 1).
        z = np.sum(np.exp(scores))
        y_probabilities = np.exp(scores) / z

        # Update weights with stochastic gradient descent
        self.W += learning_rate * (y_one_hot - y_probabilities) * x_i[None, :]
        


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.

        # Initialize weight matrix with normal distribution N(mu, sigma^2)
        mu, sigma = 0.1, 0.1
        self.W_1 = np.random.normal(mu, sigma, n_classes * n_features)
        self.W_2 = np.random.normal(mu, sigma, n_classes * n_classes)
        self.W_1 = np.reshape(self.W_1, (n_classes, n_features))
        self.W_2 = np.reshape(self.W_1, (n_classes, n_features)) # Change

        self.b_1 = 0 # Should be a vector (n_classes x n_examples)
        self.b_2 = 0 # Should be a vector (n_classes x n_examples)

        # Add biases (equal to zero)
        # np.concatenate([np.zeros((n_classes, 1)), self.W_1], axis=1) # First column of zero values (bias)???

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        self.z_1 = np.matmul(self.W_1, X.T) + self.b_1 # (n_classes x n_examples)
        self.h_1 = np.maximum(0, self.z_1) # ReLU activation, (n_classes x n_examples)

        self.z_2 = self.W_2 * self.h_1.T + self.b_2 # Should be (n_classes x n_examples)

        # Softmax 
        Z_each_example = (np.exp(self.z_2)).sum(axis=0) # (1 x n_examples)

        n_examples = np.size(self.X, 1)
        n_classes = np.size(self.W_1, 0)

        y_probabilities = np.zeroes(n_classes, n_examples) # (n_classes x n_examples)
        y_hat = np.zeroes(1, n_examples) # (1 x n_examples)

        for i in range(n_examples+1): # For each example
            y_probabilities[:,i] = np.exp(self.z_2[:,i]) / Z_each_example[0,i] # Should be (n_classes x n_examples)
            y_hat[1,i] = y_probabilities.argmax(x=0) # Should be (1 x n_examples)

        raise y_hat

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        raise NotImplementedError


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
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
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
