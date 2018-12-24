import numpy as np


def logsumexp(a, b=1.0, **kwargs):
    # a = np.clip(a, a_min=1e-16, a_max=np.inf)
    return np.log(np.sum(b*np.exp(a), **kwargs))


def logsum(a, **kwargs):
    return np.log(np.sum(a, **kwargs))


def logit(z, **kwargs):
    # z = z.clip(1e-16, 1.0-1e-16)
    # print(np.max(z), np.min(z))
    return np.exp(np.log(1.0) - np.log((1.0 + np.exp(-z, **kwargs))))


def logit_prime(z):
    return logit(z) * (1.0 - logit(z))


def softmax(x):
    z = np.exp(x) / np.exp(logsumexp(x))
    return np.clip(z, 1e-16, np.max(z))


def softmax_prime(x):
    return softmax(x)/(1.0 - softmax(x))


def tanh_prime(x):
    return 1.0 - np.square(tanh(x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return 1.0 * (x > 0)


def leaky_relu(x, epsilon=0.1):
    return np.maximum(epsilon * x, x)


def leaky_relu_prime(x, epsilon=0.1):
    gradients = 1.0 * (x > 0)
    gradients[gradients == 0] = epsilon
    return gradients
