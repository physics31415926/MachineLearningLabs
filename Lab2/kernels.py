import numpy as np


def Linear_kernel(x, y):
    return np.dot(np.transpose(x), y)


def Polynomial_kernel(x, y, p=2):
    return np.power(np.dot(np.transpose(x), y) + 1, p)


def RBF_kernel(x, y, sigma=1):
    return np.exp(-1 * np.power(np.linalg.norm(np.subtract(x, y)), 2) / (2 * np.power(sigma, 2)))
