import numpy as np
import matplotlib.pyplot as plt

from ..methods import local_mds

# Generate data
t = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
R = 1

y = R * np.sign(t) - R * np.sign(t) * np.cos(t / R)
x = -R * np.sin(t / R)
data = np.column_stack((x, y))

# Plot data
#plt.plot(x, y, color='red', linewidth=2)

# Project data
projection = local_mds(data, r=1, k=5, tau=0.5, verbose = 2)

plt.scatter(projection, t, facecolors='none', edgecolors='black')
plt.ylim([-30,30])
plt.xlabel('lambda')
plt.ylabel('rt')
plt.title('1D Projection from local_mds')
plt.show()
