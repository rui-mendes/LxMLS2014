__author__ = 'ruimendes'

import numpy as np
m = 3
n = 2
a = np.zeros([m, n])
print a

print a.shape
# (3, 2)
print a.dtype.name
# float64

a = np.zeros([m, n], dtype=int)
print a.dtype
# int64

a = np.array([[2, 3], [3, 4]])
print a
# [[2 3]
#  [3 4]]