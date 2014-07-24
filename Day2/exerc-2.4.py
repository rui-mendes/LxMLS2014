__author__ = 'ruimendes'

'''
Look at the module sequences/log_domain.py. This module implements a function logsum pair(logx, logy) to add two
numbers represented in the log-domain; it returns their sum also represented in the log-domain. The
function logsum(logv) sums all components of an array represented in the log-domain. This will be used later in our
decoding algorithms. To observe why this is important, type the following:
'''

import numpy as np

a = np.random.rand(10)
print np.log(sum(np.exp(a)))
print np.log(sum(np.exp(10*a)))
print np.log(sum(np.exp(100*a)))
print np.log(sum(np.exp(1000*a)))
# inf
from lxmls.sequences.log_domain import *
print logsum(a)
print logsum(10*a)
print logsum(100*a)
print logsum(1000*a)
