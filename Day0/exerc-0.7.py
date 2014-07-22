__author__ = 'ruimendes'

import numpy as np

i = np.arange(999)
numerator = i/1000.0


def f(x):
    return x**2

denominator = f(numerator)/1000

print denominator.sum()

