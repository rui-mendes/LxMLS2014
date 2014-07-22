__author__ = 'ruimendes'

import numpy as np
import matplotlib.pyplot as plt

a = np.arange(-5, 5, 0.01)
f_x = np.power(a, 2)
plt.plot(a, f_x)
plt.xlim(-5, 5)
plt.ylim(-5, 15)
k = np.array([-2, 0, 2])
plt.plot(k, k**2, "bo")
for i in k:
    # The equation of the tangent line to y = f(x) at the point (x1,y1):
    # ( 'm' is the gradient at (x1,y1) )
    # Y = m(x - x1) + y1
    plt.plot(a, (2*i)*a - (i**2))

plt.show()