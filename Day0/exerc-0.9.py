__author__ = 'ruimendes'

import numpy as np

print '---- Multiplication 1--------'
a = np.array([[2, 3], [3, 4]])
b = np.array([[1, 1], [1, 1]])
a_dim1, a_dim2 = a.shape
b_dim1, b_dim2 = b.shape
c = np.zeros([a_dim1, b_dim2])
for i in xrange(a_dim1):
    for j in xrange(b_dim2):
        for k in xrange(a_dim2):
            c[i, j] += a[i, k]*b[k, j]
print c

print '---- Multiplication 2.1 --------'
d = np.dot(a, b)
print d

print '---- Multiplication 2.2--------'
a = np.array([1, 2])
b = np.array([1, 1])
print np.dot(a, b)

print '---- Outer product --------'
print np.outer(a, b)

print '---- Identity Matrix--------'
I = np.eye(2)
x = np.array([2.3, 3.4])
print I
print np.dot(I, x)

print '---- Transpose Matrix --------'
A = np.array([[1, 2], [3, 4]])
print A.T