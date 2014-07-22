__author__ = 'ruimendes'

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-8, 8, 1000)
Y = (X+2)**2 - 16*np.exp(-((X-2)**2))


# derivative of the function f
def get_Y_dev(x):
    return (2*x+4)-16*(-2*x + 4)*np.exp(-((x-2)**2))


def grad_desc(start_x, eps, prec):
    """
    runs the gradient descent algorithm and returns the list of estimates
    example of use grad_desc(X, 0.01, 0.00001)
    """
    x_new = start_x
    x_old = start_x + prec * 2
    res = [x_new]
    while abs(x_old-x_new) > prec:
        x_old = x_new
        x_new = x_old - eps * get_Y_dev(x_new)
        res.append(x_new)
    return np.array(res)
