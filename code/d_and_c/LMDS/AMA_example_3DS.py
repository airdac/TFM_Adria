import numpy as np
import matplotlib.pyplot as plt

from ..methods import local_mds

# Generate data
t = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
R = 1

y = R * np.sign(t) - R * np.sign(t) * np.cos(t / R)
x = -R * np.sin(t / R)
z = (y/(2*R))**2
data = np.column_stack((x, y))

# # Plot data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(x, y, z, color='red', linewidth=2)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('3D S data')
# plt.show()

# Project data
projection = local_mds(data, r=1, k=5, tau=0.5, verbose=2)

plt.scatter(projection, t, facecolors='none', edgecolors='black')
plt.xlabel('lambda')
plt.ylabel('rt')
plt.title('2D Projection from local_mds')
plt.show()
