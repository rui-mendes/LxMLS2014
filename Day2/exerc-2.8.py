__author__ = 'ruimendes'

'''
Implement a method for performing Viterbi decoding in file sequence classification decoder.py.
'''
import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.hmm as hmmc
import warnings
warnings.filterwarnings('ignore')

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)


hmm.train_supervised(simple.train, smoothing=0.1)
y_pred, score = hmm.viterbi_decode(simple.test.seq_list[0])
print "Viterbi decoding Prediction test 0 with smoothing:\n", y_pred, score
# walk/rainy walk/rainy shop/sunny clean/sunny -6.02050124698
print "Truth test 0:\n", simple.test.seq_list[0]
# walk/rainy walk/sunny shop/sunny clean/sunny
y_pred, score = hmm.viterbi_decode(simple.test.seq_list[1])
print "\n\nViterbi decoding Prediction test 1 with smoothing:\n", y_pred, score
# clean/sunny walk/sunny tennis/sunny walk/sunny -11.713974074
print "Truth test 1\n:", simple.test.seq_list[1]
# clean/sunny walk/sunny tennis/sunny walk/sunny
