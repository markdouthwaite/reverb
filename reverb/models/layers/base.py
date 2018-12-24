import abc
import apogee as ap
from ..activations import activations, Activation


class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    @ap.random.seeded
    def __init__(self, n, m, activation="logistic", state=None, shift=-0.5, scale=2.0, *args, **kwargs):
        self.state = state
        self.weights = self._init_weights(n, m, state=state, shift=shift, scale=scale, *args, **kwargs)
        self.func = activation if issubclass(type(activation), Activation) else activations.get(activation)

    def __call__(self, *args, **kwargs):
        # bit ugly!
        if "bias" not in kwargs:
            return self.activate(*args, **kwargs)
        else:
            return self.activate(*args, **kwargs)

    @abc.abstractmethod
    def activate(self, inputs, **kwargs):
        pass

    @property
    def units(self):
        return self.shape[0]

    @property
    def shape(self):
        return self.weights.shape

    @staticmethod
    def _init_weights(n, m, *args, **kwargs):
        if kwargs.get("bias", True):
            m += 1

        if "bias" in kwargs:
            del kwargs["bias"]

        return ap.random.random_array(n, m, *args, **kwargs)


