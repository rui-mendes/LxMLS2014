__author__ = 'ruimendes'

'''
Run the posterior decode on the first test sequence, and evaluate it.
'''

import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.hmm as hmmc
import warnings
warnings.filterwarnings('ignore')

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
state_posteriors, _, _ = hmm.compute_posteriors(initial_scores, transition_scores, final_scores, emission_scores)

y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0:", y_pred
print "Truth test 0:", simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "\nPrediction test 1:", y_pred
print "Truth test 1:", simple.test.seq_list[1]

'''
What is wrong? Note the observations for the second test sequence: the observation tennis was never seen at
training time, so the probability for it will be zero (no matter what state). This will make all possible state sequences have
zero probability. As seen in the previous lecture, this is a problem with generative models, which can be corrected using
smoothing (among other options).
Change the train supervised method to add smoothing.
Try, for example, adding 0.1 to all the counts, and repeating this exercise with that smoothing. What do you observe?
'''

print "\n######################\n"
hmm.train_supervised(simple.train, smoothing=0.1)
y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0 with smoothing:", y_pred
# walk/rainy walk/rainy shop/sunny clean/sunny
print "Truth test 0:", simple.test.seq_list[0]
# walk/rainy walk/sunny shop/sunny clean/sunny
y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "\nPrediction test 1 with smoothing:", y_pred
# clean/sunny walk/sunny tennis/sunny walk/sunny
print "Truth test 1:", simple.test.seq_list[1]
# clean/sunny walk/sunny tennis/sunny walk/sunny


