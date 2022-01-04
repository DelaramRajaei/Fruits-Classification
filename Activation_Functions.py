import math


def sigmoid(z):
    try:
        res = 1 / (1 + math.exp(-z))
    except OverflowError:
        res = 0
    return res


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    if z >= 0:
        return z
    else:
        return 0


def relu_derivative(z):
    if z >= 0:
        return 1
    else:
        return 0

