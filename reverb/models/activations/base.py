import numpy as np


class Activation(object):
    def __init__(self, f: callable, p: callable):
        self.func = f
        self.prime = p
        self.z = None
        self.p = None

    def __call__(self, *args, bias: bool=True, prime=True, **kwargs):
        pre_activation = self.process(*args, **kwargs)
        act = self.func(pre_activation)
        if bias:
            self.z = np.concatenate((np.asarray([1]), act))
        else:
            self.z = act
        if prime:
            self.p = self.prime(pre_activation)
        return self.z, self.p

    def process(self, x: np.ndarray, **kwargs):
        # semi-redundant for now
        return x
