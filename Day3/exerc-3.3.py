__author__ = 'ruimendes'

'''
Implement the structured perceptron algorithm
To do this, edit file sequences/structured perceptron.py and implement the function
def perceptron_update(self, sequence)
'''
import lxmls.sequences.crf_online as crfo
import lxmls.sequences.structured_perceptron as spc
import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.extended_feature as exfc

print "Perceptron Exercise"
corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll", max_sent_len=10, max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("../data/test-23.conll", max_sent_len=10, max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("../data/dev-22.conll", max_sent_len=10, max_nr_sent=1000)

feature_mapper = idfc.IDFeatures(train_seq)
feature_mapper.build_features()

sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
sp.num_epochs = 20
sp.train_supervised(train_seq)

pred_train = sp.viterbi_decode_corpus(train_seq)
pred_dev = sp.viterbi_decode_corpus(dev_seq)
pred_test = sp.viterbi_decode_corpus(test_seq)
eval_train = sp.evaluate_corpus(train_seq, pred_train)
eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
eval_test = sp.evaluate_corpus(test_seq, pred_test)

# Confusion_matrix calculation
import lxmls.sequences.confusion_matrix as cm
import matplotlib.pyplot as plt
confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, pred_test, len(corpus.tag_dict),
                                             sp.get_num_states())

cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict,
                            xrange(sp.get_num_states()), 'Confusion matrix')