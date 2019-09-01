import numpy as np
from random import shuffle
import math


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  number_class = W.shape[1]  # number of classes to classify 10
  number_values = X.shape[0]  # number of instances 500

  # cumpute the loss
  for i in range(number_values):
    scores = X[i].dot(W)
    correct_score = scores[y[i]]
    exp_scores = pow(math.e, correct_score).sum()

    loss += - correct_score + np.log(exp_scores) + scores.max()

    for f in range(number_class):
      # dW[:, f] += X[i] * correct_score / exp_scores
      dW[:, f] += np.exp(scores[f]) / np.exp(scores).sum() * X[i, :]

  loss /= number_values
  dW /= dW
  loss += reg * np.sum(W * W)

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)
  number_class = W.shape[1]
  number_values = X.shape[0]

  scores = X.dot(W)

  correct_score = scores[range(number_values), y]

  maximun_score = scores.max(axis=1, keepdims=True)

  scores -= maximun_score

  loss = - correct_score.sum() + maximun_score.sum() + \
      np.log(np.exp(scores).sum(axis=1)).sum()

  loss /= number_values
  loss += reg * np.sum(W * W)

  softmax = (np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1, 1))
  softmax[range(number_values), y] -= 1

  W = X.T.dot(softmax)
  dW /= number_values
  dW += 2 * reg * W

  return loss, dW
