__author__ = 'ruimendes'

'''
Convince yourself that the score of a path in the trellis (summing over the scores above) is equivalent to the
log-probability log P ( X = x, Y = y ) , as defined in Eq. 2.2. Use the given function compute scores on the first training
sequence and confirm that the values are correct. You should get the same values as presented below.
'''
import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.hmm as hmmc

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
print 'Initial scores: ', initial_scores
'''
    rainy       sunny
[-0.40546511 -1.09861229]

Calculation:
[ ln(2/3)   ln(1/3) ]

where ln(2/3) is the number of rainy state was preceded by start signal
where ln(1/3) is the number of sunny state was preceded by start signal
'''

print 'Transition scores: ', transition_scores
'''
    rainy       sunny
[[[-0.69314718  -inf]
  [-0.69314718  -0.47000363]]

 [[-0.69314718  -inf]
  [-0.69314718  -0.47000363]]

 [[-0.69314718  -inf]
  [-0.69314718  -0.47000363]]]

  Calculation:
[[[ (*) ln(0.5)   ln(0) ]
 [ (**) ln(0.5)   ln(5/8)]]
 ...]]]

where (*) ln(0.5) it's the log of the probability of going from a rainy state to another rainy state
where ln(0) it's the log of the probability of going from a sunny state to a rainy state
where (**) ln(0.5) it's the log of the probability of going from a rainy state to a sunny state
where ln(5/8) it's the log of the probability of going from a sunny state to another sunny state
'''
print 'Final scores: ', final_scores
'''
  rainy   sunny
[ -inf -0.98082925]

Calculation:
[ ln(0)   ln(3/8) ]

where ln(0) it's the log of the probability of being in a stop state coming from a rainy state
where ln(3/8) it's the log of the probability of being in a stop state coming from a sunny state
'''
