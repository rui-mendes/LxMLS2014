__author__ = 'ruimendes'

'''
Run the provided forward-backward algorithm on the first train sequence. Observe that both the forward
and the backward passes give the same log-likelihood.
'''

import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.hmm as hmmc

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])

log_likelihood, forward = hmm.decoder.run_forward(initial_scores, transition_scores,
                                                  final_scores, emission_scores)

print 'Log-Likelihood =', log_likelihood
# Log-Likelihood = -5.06823232601

log_likelihood, backward = hmm.decoder.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
print 'Log-Likelihood =', log_likelihood
# Log-Likelihood = -5.06823232601