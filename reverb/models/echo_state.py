import numpy as np
from .layers import Layer, Readout, Reservoir


class EchoStateNetwork(object):
    def __init__(self, n: int, m: int = 1, noise: float = 1e-3, activation: str = "tanh", alpha: float = 0.95,
                 seed=None, **kwargs):
        """
        A 'standard' sequential implementation of an ESN model.

        Parameters
        ----------
        n: int
        m: int
        noise: float
        activation: str
        alpha: float
            The spectral radius of the ESN.

        Examples
        --------

        """
        state = np.random.RandomState(seed)
        self.input = Layer(n, m, activation="null", bias=False, state=state)
        self.reservoir = Reservoir(n, bias=False, noise=noise, activation=activation, alpha=alpha, state=state,
                                   **kwargs)
        self.feedback = Layer(n, m, bias=False, activation="null", state=state)
        self.readout = Readout(m, n + 1, bias=False, activation="null", state=state)
        self._y = None

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        for i in range(1, len(x)):
            self.reservoir(self.input(x[i]), self.feedback(y[i - 1]))
        self._y = y[-1]
        return self.readout.optimise(self.reservoir.x, x[1:], y[1:], **kwargs).squeeze()

    def predict(self, x, y=None, y0=None):
        self.reservoir.purge()
        for i in range(len(x)):
            if i == 0:
                self.reservoir(self.input(x[i]), self.feedback(y0 or self._y))
            elif y is None:
                z = self.readout(self.reservoir.x[-1], x[i - 1])
                self.reservoir(self.input(x[i]), self.feedback(z[1:]))
            else:
                self.reservoir(self.input(x[i]), self.feedback(y[i - 1]))
        if y is not None:
            self._y = y[-1]
        else:
            self._y = self.readout(self.reservoir.x[-1], x[i - 1])
        return self.readout.batch(self.reservoir.x[-len(x):], x).squeeze()
