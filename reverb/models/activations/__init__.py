from .base import Activation
from .funcs import logit, logit_prime, leaky_relu, leaky_relu_prime, logsum, \
        logsumexp, relu, relu_prime, softmax, softmax_prime, tanh_prime, tanh

activations = {
        "null": Activation(lambda x: x, lambda x: x),
        "logistic": Activation(logit, logit_prime),
        "tanh": Activation(tanh, tanh_prime),
        "softmax": Activation(softmax, softmax_prime),
        "relu": Activation(relu, relu_prime),
        "leaky_relu": Activation(leaky_relu, leaky_relu_prime)
    }


