from reverb.models import EchoStateNetwork
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, np.pi*4, 1000).reshape(-1, 1)
y = np.sin(x).reshape(-1, 1)

tx, ty = x[:800], y[:800]
vx, vy = x[800:], y[800:]

esn = EchoStateNetwork(n=500)
esn.fit(tx, ty)

plt.plot(esn.predict(vx, vy))
plt.plot(vy)
plt.show()
