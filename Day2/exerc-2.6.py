__author__ = 'ruimendes'

'''
Compute the node posteriors for the first training sequence (use the provided compute posteriors function), and look
at the output. Note that the state posteriors are a proper probability distribution (the lines of the result sum to 1).
'''

import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.hmm as hmmc

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
state_posteriors, _, _ = hmm.compute_posteriors(initial_scores, transition_scores, final_scores, emission_scores)
print state_posteriors
'''
First sentence (simple.train.seq_list[0])
        rainy       sunny
walk [[ 0.95738152 0.04261848]
walk [ 0.75281282 0.24718718]
shop [ 0.26184794 0.73815206]
clean [ 0.         1.        ]]
'''
