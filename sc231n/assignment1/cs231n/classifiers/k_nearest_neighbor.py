import numpy as np
from scipy.spatial import distance


class KNearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)
      return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point. """

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        dists[i, j] = np.linalg.norm(X[i] - self.X_train[j])

    # dists[i, j] = self.X_train[np.newaxis, :, :] - X[:, np.newaxis, :]

    #####################################################################
    # TODO:                                                             #
    # Compute the l2 distance between the ith test point and the jth    #
    # training point, and store the result in dists[i, j]. You should   #
    # not use a loop over dimension.                                    #
    #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #
      # dists[i, :] = np.sum(X[i] - self.X_train)

      # mport time
      # tic = time.time()
      # toc = time.time()
      # print("time", toc - tic)
      dists[i] = np.linalg.norm(self.X_train - np.array([X[i]] * num_train), axis=1)

      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################

      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # first variant, but didn/t work - broadcasting error

    # dists_no_loop_0 = np.sum(X * X) + np.sum(self.X_train * self.X_train, axis=1) - \
    #X.dot(self.X_train.T) * 2

    # print(dists_no_loop_0.shape)

    # + np.array((np.sum(self.X_train * self.X_train, axis=1) * X.shape[0]).transpose())

    # - X.dot(self.X_train.T) * 2

    # second, not enought memory

    # dists_no_loop_1 = self.X_train[np.newaxis, :, :] - X[:, np.newaxis, :]

    # using scipy
    # from scipy.spatial.distance import cdist

    # dists_no_loop_2=scipy.spatial.distance.cdist(X, self.X_train.T)

    #''' from git '''

    X_train_2 = self.X_train * self.X_train
    #print(X_train_2.shape)

    X_train_2 = np.sum(X_train_2, axis=1)
    #print(X_train_2.shape)

    X_train_2_repeat = np.array([X_train_2] * X.shape[0])
    #print(X_train_2_repeat.shape)

    X_2 = X * X
    #print(X_2.shape)
    X_2 = np.sum(X_2, axis=1)
    #print(X_2.shape)

    X_2_repeat = np.array([X_2] * self.X_train.shape[0]).transpose()
    #print(X_2_repeat.shape)

    X_dot_X_train = X.dot(self.X_train.T)
    #print(X_dot_X_train.shape)

    dists = X_train_2_repeat + X_2_repeat - 2 * X_dot_X_train
    #print(dists.shape)

    #dists = np.sqrt(dists)

    return np.sqrt(dists)

    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []

      # min_distance = np.argsort(dists[i])

      # closest_y = self.y_train[min_distance]

      closest_y = self.y_train[dists[i].argsort()[:k]]

      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      from collections import Counter
      y_pred[i] = Counter(closest_y).most_common(1)[0][0]

      # y_pred[i] = np.bincount(closest_y).argmax()

      #########################################################################
      #                           END OF YOUR CODE                            #
      #########################################################################
    return y_pred
