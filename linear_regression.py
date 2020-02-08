"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


def _error(y, w):
    """ Simple error """
    return np.subtract(y, w)


###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    temp = np.dot(X, w)
    err = np.mean(np.abs(_error(y, temp)))
    return err



###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################
    temp = X.T
    result = np.dot(temp, X)
    result = np.linalg.inv(result)
    result = np.dot(result, temp)
    w = np.dot(result, y)
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    X_X_T = np.dot(X.T, X)
    ev = 0
    while ev < (10**-5):
        ev = np.min((np.linalg.eig(X_X_T)[0]))
        if ev < (10**-5):
            X_X_T = X_X_T + (10**-1) * np.identity(X_X_T.shape[0])

    w = np.dot(np.dot(np.linalg.inv(X_X_T), X.T), y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################
    w = None
    X_X_T = np.dot(X.T, X)
    X_X_T  += lambd * np.identity(X_X_T.shape[0])
    w = np.dot(np.dot(np.linalg.inv(X_X_T), X.T), y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    bestlambda = None
    err = 1

    for v in range(-19,20):
        if v>=0:
            val = float("1e+"+str(v))
        else:
            val = float("1e"+str(v))
        w = regularized_linear_regression(Xtrain,ytrain, val)
        error = mean_absolute_error(w, Xval,yval)
        if err > error:
            err = error
            bestlambda = val
    return bestlambda



###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    X_temp = X[:]
    for i in range(2, power + 1):
        x = np.power(X, i)
        for col in range(0, x.shape[1]):
            X_temp = np.insert(X_temp, X_temp.shape[1], x[:, col], axis=1)

    return X_temp


