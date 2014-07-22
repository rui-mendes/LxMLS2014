__author__ = 'ruimendes'

import numpy as np
A = np.arange(100)
# These two lines do exactly the same thing
print np.mean(A)
print A.mean()
C = np.cos(A)
print C.ptp()

"""
Type:       function
String Form:<function ptp at 0x7fbabb23ec08>
File:       /usr/lib/python2.7/dist-packages/numpy/core/fromnumeric.py
Definition: np.ptp(a, axis=None, out=None)
Docstring:
Range of values (maximum - minimum) along an axis.

The name of the function comes from the acronym for 'peak to peak'.

Parameters
----------
a : array_like
Input values.
axis : int, optional
Axis along which to find the peaks.  By default, flatten the
array.
out : array_like
Alternative output array in which to place the result. It must
have the same shape and buffer length as the expected output,
but the type of the output values will be cast if necessary.

Returns
-------
ptp : ndarray
A new array holding the result, unless `out` was
specified, in which case a reference to `out` is returned.

Examples
--------
>>> x = np.arange(4).reshape((2,2))
>>> x
array([[0, 1],
       [2, 3]])

>>> np.ptp(x, axis=0)
array([2, 2])

>>> np.ptp(x, axis=1)
array([1, 1])
"""