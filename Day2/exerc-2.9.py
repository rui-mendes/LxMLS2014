__author__ = 'ruimendes'

import lxmls.sequences.hmm as hmmc
import warnings
warnings.filterwarnings('ignore')
import lxmls.readers.pos_corpus as pcc


'''
The first step is to load the corpus. We will start by loading 1000 sentences for training and 1000 sentences
both for development and testing. Then we train the HMM model by maximum likelihood estimation.
'''
corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,
                                                max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("../data/test-23.conll",max_sent_len=15,
                                               max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("../data/dev-22.conll",max_sent_len=15,
                                              max_nr_sent=1000)
hmm = hmmc.HMM(corpus.word_dict, corpus.tag_dict)
hmm.train_supervised(train_seq)
hmm.print_transition_matrix()

'''
Test the model using both posterior decoding and Viterbi decoding on both the train and test set, using the
methods in class HMM:
'''
viterbi_pred_train = hmm.viterbi_decode_corpus(train_seq)
posterior_pred_train = hmm.posterior_decode_corpus(train_seq)
eval_viterbi_train = hmm.evaluate_corpus(train_seq, viterbi_pred_train)
eval_posterior_train = hmm.evaluate_corpus(train_seq, posterior_pred_train)
print "Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f" % (eval_posterior_train, eval_viterbi_train)
# Train Set Accuracy: Posterior Decode 0.985, Viterbi Decode: 0.985
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
eval_viterbi_test = hmm.evaluate_corpus(test_seq,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq,posterior_pred_test)
print "Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f" % (eval_posterior_test,eval_viterbi_test)
# Test Set Accuracy: Posterior Decode 0.350, Viterbi Decode: 0.509

'''
What do you observe? Remake the previous exercise but now train the HMM using smoothing. Try different values
(0,0.1,0.01,1) and report the results on the train and development set. (Use function pick best smoothing).
'''
best_smoothing = hmm.pick_best_smoothing(train_seq, dev_seq, [10, 1, 0.1, 0])
# Smoothing 10.000000 -- Train Set Accuracy: Posterior Decode 0.731, Viterbi Decode: 0.691
# Smoothing 10.000000 -- Test Set Accuracy: Posterior Decode 0.712, Viterbi Decode: 0.675
# Smoothing 1.000000 -- Train Set Accuracy: Posterior Decode 0.887, Viterbi Decode: 0.865
# Smoothing 1.000000 -- Test Set Accuracy: Posterior Decode 0.818, Viterbi Decode: 0.792
# Smoothing 0.100000 -- Train Set Accuracy: Posterior Decode 0.968, Viterbi Decode: 0.965
# Smoothing 0.100000 -- Test Set Accuracy: Posterior Decode 0.851, Viterbi Decode: 0.842
# Smoothing 0.000000 -- Train Set Accuracy: Posterior Decode 0.985, Viterbi Decode: 0.985
# Smoothing 0.000000 -- Test Set Accuracy: Posterior Decode 0.370, Viterbi Decode: 0.526
hmm.train_supervised(train_seq, smoothing=best_smoothing)
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
eval_viterbi_test = hmm.evaluate_corpus(test_seq, viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq, posterior_pred_test)
print "Best Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, ViterbiDecode: %.3f" % (best_smoothing, eval_posterior_test,eval_viterbi_test)
# Best Smoothing 0.100000 -- Test Set Accuracy: Posterior Decode 0.837, Viterbi Decode: 0.827

'''
Perform some error analysis to understand were the errors are coming from. You can start by visualizing the confusion
matrix (true tags vs predicted tags).
'''
import lxmls.sequences.confusion_matrix as cm
import matplotlib.pyplot as plt
confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, viterbi_pred_test, len(corpus.tag_dict), hmm.get_num_states())
cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict, xrange(hmm.get_num_states()), 'Confusion matrix')
plt.show()