import numpy as np
from scipy.special import expit


def sigmoid(x):
    """
    A numpy implementation of sigmoid function
    """
    # return 1.0 / (1. + np.exp(-x))
    return expit(x)


def derivative_sigmoid(x):
    """
    Returns the derivative of sigmoid function
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def softmax(x):
    x_max = np.amax(x, axis=1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
