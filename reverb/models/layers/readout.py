from .dense import Layer
import numpy as np


class Readout(Layer):
    # add NeuralModel optimisation.
    def __init__(self, n, m, **kwargs):
        super().__init__(n, m, **kwargs)

    def activate(self, *inputs, **kwargs):
        inputs = np.concatenate(inputs)
        return super().activate(inputs, **kwargs)

    def optimise(self, *args, mode="pinv", **kwargs):
        if mode == "pinv":
            return self._pinv(*args, **kwargs)
        elif mode == "rr":
            return self._ridge(*args, **kwargs)

    def _pinv(self, states, inputs, outputs, burn=10):
        states_ = np.hstack((states, inputs))
        states_[np.isinf(states_)] = 0.0
        states_[np.isnan(states_)] = 0.0
        self.weights = np.linalg.pinv(states_[burn:]).dot(outputs[burn:]).T
        return states_.dot(self.weights.T)

    def _ridge(self, states, inputs, outputs, burn=10, r=1e-6):
        states_ = np.hstack((states, inputs))
        states_[np.isinf(states_)] = 0.0
        states_[np.isnan(states_)] = 0.0
        states = states_[burn:]
        trans = states.T
        self.weights = np.dot(np.dot(outputs[burn:].T, states),
                              np.linalg.inv(np.dot(trans, states))
                              + r * np.eye(len(states[0])))

        return states_.dot(self.weights.T)

    def batch(self, states, inputs):
        states_ = np.hstack((states, inputs))
        return states_.dot(self.weights.T)
