__author__ = 'ruimendes'

'''
In this exercise you will train a CRF using different feature sets for Part-of-Speech Tagging.
'''

import lxmls.sequences.crf_online as crfo
import lxmls.sequences.structured_perceptron as spc
import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.extended_feature as exfc

print "CRF Exercise"
corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll", max_sent_len=10, max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("../data/test-23.conll", max_sent_len=10, max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("../data/dev-22.conll", max_sent_len=10, max_nr_sent=1000)
feature_mapper = idfc.IDFeatures(train_seq)
feature_mapper.build_features()

crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
crf_online.num_epochs = 20
crf_online.train_supervised(train_seq)

'''
You will receive feedback when each epoch is finished, note that running the 20 epochs might take a while.
After training is done, evaluate the learned model on the training, development and test sets.
'''

pred_train = crf_online.viterbi_decode_corpus(train_seq)
pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
pred_test = crf_online.viterbi_decode_corpus(test_seq)
eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
eval_test = crf_online.evaluate_corpus(test_seq, pred_test)

print "CRF - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train, eval_dev, eval_test)

# CRF -
# ID Features Accuracy Train: 0.949 Dev: 0.846 Test: 0.858

# Confusion_matrix calculation
import lxmls.sequences.confusion_matrix as cm
import matplotlib.pyplot as plt
confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, pred_test, len(corpus.tag_dict),
                                             crf_online.get_num_states())

cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict,
                            xrange(crf_online.get_num_states()), 'Confusion matrix')