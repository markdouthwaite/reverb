import numpy as np
from .base import BaseLayer


class Layer(BaseLayer):
    def __init__(self, n, m, *args, **kwargs):
        super().__init__(n, m, *args, **kwargs)
        self.x = np.zeros(m)  # input
        self.z = np.zeros(m)  # remove
        self.p = np.zeros(m)  # remove

    def activate(self, inputs, **kwargs):
        self.x = inputs
        self.z, self.p = self.func(self.weights.dot(inputs), **kwargs)  # still dislike.
        return self.z
