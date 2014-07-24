__author__ = 'ruimendes'

import lxmls.readers.simple_sequence as ssr

simple = ssr.SimpleSequence()
print simple.train
print simple.test

for sequence in simple.train.seq_list:
    print sequence

for sequence in simple.train.seq_list:
    print sequence.x

for sequence in simple.train.seq_list:
    print sequence.y



