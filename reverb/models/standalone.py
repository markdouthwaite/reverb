import apogee as ap
import numpy as np


class MonolithESN(object):
    def __init__(self, n: int=1, m: int=50, k: int=1,
                 sr: float=0.95, shift: float=-0.5, scale: float=2.0,
                 sp: float=0.0, f: callable=ap.numeric.tanh,
                 noise: float=1e-4, leak: float=1.0, seed: int=None):
        """
        A standalone implementation of an ESN with a single reservoir.
        
        Parameters
        ----------
        n: int
            Number of input units.
        m: int
            Number of reservoir units.
        k: int
            Number of output units.
        sr: float
            Scaling factor for the spectral radius of the reservoir. 
        shift: float
            Shift factor for *all* layers. Modify individual layers separately.
        scale: float
            Scaling factor for *all* layers. Modify individual layers separately.
        sp: float
            
        f: callable
            The activation function for the ESN.
        noise: float
            Noise to apply during optimisation (if applicable).
        leak: float
            Leak term for reservoir activations.
        seed: int
            Seed for random state of the ESN.
            
        See Also
        --------
        esn.EchoStateModel
        
        """

        self.output_ = None
        self._leak = leak
        self._seeds = ap.random.random_state(seed).randint(ap.MAX_SEED, size=(3,))
        self._states = {
                "training": None,
                "input": None,
                "output": None,
                "reservoir": None
            }

        self._weights = {
                "input": ap.random.random_array(m, n, shift=shift, scale=scale, seed=self._seeds[0]),
                "output": None,
                "feedback": ap.random.random_array(m, k, shift=shift, seed=self._seeds[1]),
                "reservoir": ap.random.random_array(m, m, alpha=sr, shift=shift, seed=self._seeds[2], sparsity=sp)
            }

        self.f = f
        self.states = None
        self.noise = noise

    def fit(self, x: np.ndarray, y: np.ndarray, burn: int=0,
            feedback: bool=True, opt: str="pinv", **kwargs):

        x = x.reshape((len(x), -1))
        y = y.reshape((len(y), -1))
        self.states = np.zeros((x.shape[0], self.m))

        self.state = self.states[0]
        self.example = self.example or np.zeros_like(y[0])
        for i in range(1, len(self.states)):
            self.states[i] += self._compute_activation(x[i], None, feedback=feedback)
            self.state = self.states[i]
            self.input = x[i]
            self.example = y[i]

        return self._optimise(x, y, burn, mode=opt, **kwargs)

    def _compute_activation(self, x: np.ndarray, y: np.ndarray=None, feedback: bool=True):
        z1 = self.wi.dot(x)
        z2 = self.wr.dot(self.state)

        if feedback:
            z3 = self.wf.dot(self.example)
        else:
            z3 = np.zeros_like(z2)
        new = self._leak * (self.f(z1+z2+z3) + (np.random.rand(z1.shape[0])-0.5)*self.noise)
        return ((1.0 - self._leak) * self.state) + new

    def predict(self, x: np.ndarray, y: np.ndarray=None):
        n = x.shape[0]
        states = np.vstack([self.state, np.zeros((n, self.m))])
        output = np.vstack([self.example, np.zeros_like(x)])
        inputs = np.vstack([self.input, x])

        for i in range(x.shape[0]):
            states[i+1] += self._compute_activation(inputs[i+1], output[i], feedback=True)
            z = np.concatenate((ap.arr1d(states[i+1]), ap.arr1d(inputs[i+1])))
            output[i+1] = self.wo.dot(z)
            self.state = states[i+1]
            if y is not None:
                self.example = output[i+1]
            else:
                self.example = output[i + 1]
            self.input = inputs[i+1]
        return output[1:]

    def _optimise(self, x: np.ndarray, y: np.ndarray,
                  burn: int, mode: str="pinv", r: float=1e-6, **kwargs):
        states_ = np.hstack((self.states, x))
        states_[np.isinf(states_)] = 0.0
        states_[np.isnan(states_)] = 0.0

        if mode == "rr":
            ap.tools.warn("rr is temporarily disabled, switching to 'pinv'")
            o = np.linalg.pinv(states_[burn:])
            wo = o.dot(y[burn:]).T
            self._weights["output"] = wo
            # y = y[burn:]
            # states = states_[burn:]
            # states_t = states.T
            # wo = np.dot(np.dot(y.T, states), np.linalg.inv(np.dot(states_t, states)) + r * np.eye(self.inputs+self.m))
            # self._weights["output"] = wo
        else:
            o = np.linalg.pinv(states_[burn:])
            wo = o.dot(y[burn:]).T
            self._weights["output"] = wo
        return states_.dot(wo.T)

    def reconfigure(self, key: str, *args, **kwargs):
        if key in self._weights:
            self._weights[key] = ap.random.random_array(*args, **kwargs)
        else:
            raise KeyError("No weight matrix exists with key '{0}'".format(key))

    @property
    def input(self):
        return self._states["input"]

    @input.setter
    def input(self, value):
        self._states["input"] = value

    @property
    def example(self):
        return self._states["training"]

    @example.setter
    def example(self, value):
        self._states["training"] = value

    @property
    def state(self):
        return self._states["reservoir"]

    @state.setter
    def state(self, value):
        self._states["reservoir"] = value

    @property
    def wi(self):
        return self._weights["input"]

    @property
    def wo(self):
        return self._weights["output"]

    @property
    def wr(self):
        return self._weights["reservoir"]

    @property
    def wf(self):
        return self._weights["feedback"]

    @property
    def m(self):
        """
        Get the number of units in the reservoir.

        """

        return self.wr.shape[0]

    @property
    def inputs(self):
        return self._weights["input"].shape[1]

    def previous(self, key):
        return self._states[key]
