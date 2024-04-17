import numpy as np
data = np.load("test-data.npy")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(data.shape[1])
Y = np.arange(data.shape[0])
X, Y = np.meshgrid(X, Y)
Z = data

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()