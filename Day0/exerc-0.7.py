__author__ = 'ruimendes'

import numpy as np

i = np.arange(999)
numerator = i/1000.0


def f(x):
    return x**2

denominator = f(numerator)/1000

result = denominator.sum()
realValue = 1.0/3.0

approximation = realValue - result

print 'RESULT: ' + str(result)
print 'REAL VALUE: ' + str(realValue)
print 'APPROXIMATION: ' + str(approximation)

