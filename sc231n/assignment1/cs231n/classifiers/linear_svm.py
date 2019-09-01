import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]  # size of the classes
  num_train = X.shape[0]  # number of train set
  loss = 0.0  # initial loss
  num_class = 0  # initial number of classe that didn't meet the desired marrgin
  num_scores = []  # zero list to store scores

  for i in range(num_train):
    scores = X[i].dot(W)  # scores of the i of X with W
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:  # if
        continue
      margin = scores[j] - correct_class_score + 1  # note delta = 1
      num_scores.append(scores[j])  # collect all the accounted scores

      if margin > 0:
        loss += margin
        # num_class += 1  # number of classes that not meet the margin

  for i in num_scores:
    if i < 0:
      count = num_scores.count(i)

  dW += count * W * reg

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized_new(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  yi_scores = scores[np.arange(scores.shape[0]), y]  # http://stackoverflow.com/a/23435843/459241
  margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
  margins[np.arange(num_train), y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += reg * np.sum(W * W)

  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  # Average
  dW /= num_train

  # Regularize
  dW += reg * W

  return loss, dW


def svm_loss_naive_new(W, X, y, reg):

  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i, :].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i, :]
        dW[:, j] += X[i, :]

  # Averaging over all examples
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += reg * W * 2

  return loss, dW
