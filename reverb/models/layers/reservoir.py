from .base import BaseLayer
import apogee as ap
from apogee.stream import Signal


class Reservoir(BaseLayer):
    def __init__(self, n, noise=0.0, *args, **kwargs):
        super().__init__(n, n, *args, **kwargs)
        self.noise = noise
        self.matrix = ap.zeros((n, ))
        self.cache = Signal()

    def activate(self, *inputs, **kwargs):
        z = inputs[0][1:]
        if len(inputs) > 0:
            z += inputs[1][1:]

        z += self.weights.dot(self.matrix).squeeze()
        if self.noise > 0.0:
            self.matrix += self._noise()
        self.matrix = self.func(z, prime=False, bias=False)[0]
        self.cache.update(self.matrix)
        return self.matrix

    def purge(self, **kwargs):
        self.cache.purge(**kwargs)

    @property
    def theta(self):
        return self.x

    @property
    def x(self):
        return self.cache.x

    def _noise(self):
        return self.noise * (self.state.rand(self.units) - 0.5)
