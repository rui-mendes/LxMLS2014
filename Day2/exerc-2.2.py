__author__ = 'ruimendes'

'''
The provided function train supervised from the hmm.py file implements the above parameter estimates.
Run this function given the simple dataset above and look at the estimated probabilities. Are they correct? You can also
check the variables ending in counts instead of probs to see the raw counts (for example, typing hmm.initial_counts
will show you the raw counts of initial states). How are the counts related to the probabilities?
'''

import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.hmm as hmmc

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)
print "Initial Counts:\n", hmm.initial_counts
print "Initial Probabilities:\n", hmm.initial_probs

print "\nTransition Counts:\n", hmm.transition_counts
'''
Transition Counts: (transiçoes de estados)
        rainy sunny
 rainy [[ 2.  0.]
 sunny [ 2.  5.]]
'''
print "Transition Probabilities:\n", hmm.transition_probs

print "\nFinal Counts:\n", hmm.final_counts
print "Final Probabilities:\n", hmm.final_probs

print "\nEmission Counts\n", hmm.emission_counts
'''
Emission Counts (estados e acções)
        rainy sunny
 walk   [[ 3.  2.]
 shop   [ 1.  3.]
 clean  [ 0.  3.]
 tennis [ 0.  0.]]
'''
print "Emission Probabilities\n", hmm.emission_probs



