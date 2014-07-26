__author__ = 'ruimendes'
# -*- coding: utf-8 -*-

'''
In this exercise you are going to experiment with arc-factored non-projective dependency parsers.
The CoNLL-X and CoNLL 2008 shared task datasets (Buchholz and Marsi, 2006; Surdeanu et al., 2008) contain
dependency treebanks for 14 languages. In this lab, we are going to experiment with the Portuguese and English datasets.
We preprocessed those datasets to exclude all sentences with more than 15 words; this yielded the files:
• data/deppars/portuguese train.conll,
• data/deppars/portuguese test.conll,
• data/deppars/english train.conll,
• data/deppars/english test.conll.
'''

exerc = 6     # 2, 3, 4, 5, 6

# ------------------------------- 1 ------------------------------------- #
'''
1 - After importing all the necessary libraries, load the Portuguese dataset:
'''
import sys
sys.path.append('.')
import lxmls.parsing.dependency_parser as depp
dp = depp.DependencyParser()


# print len(dp.features)
'''
PORTUGUESE
Observe the statistics which are shown. How many features are there in total?
Number of sentences: 3029
Number of tokens: 25015
Number of words: 7621
Number of pos: 16
Number of features: 142

'''


# ------------------------------- 2 ------------------------------------- #
'''
2 - In the default configuration, only the basic features are enabled. The total number of features is the quantity
observed in the previous question. With this configuration, train the parser by running 10 epochs of the structured
perceptron algorithm:
'''

if exerc == 2:
    print '#### Exerc. 2 #####'
    print '****** portuguese *************'
    dp.read_data("portuguese")
    dp.train_perceptron(10)
    dp.test()

'''
What is the accuracy obtained in the test set? (Note: the shown accuracy is the fraction of words whose parent was
correctly predicted.)
R: Test accuracy (109 test instances): 0.495210727969
'''

# --------------------------------- 3 ----------------------------------- #
'''
3 - Repeat the previous exercise by subsequently enabling the lexical, distance and contextual features:
For each configuration, write down the number of features and test set accuracies. Observe the improvements
obtained when more features were added.
Feel free to engineer new features!
'''

if exerc == 3:
    print '#### Exerc. 3 #####'

    print '------ use_lexical=TRUE ------------'
    dp.features.use_lexical = True
    dp.read_data("portuguese")
    dp.train_perceptron(10)
    dp.test()
    '''
    Number of sentences: 3029
    Number of tokens: 25015
    Number of words: 7621
    Number of pos: 16
    Number of features: 46216
    Test accuracy (109 test instances): 0.57662835249
    '''

    print '------ use_distance=TRUE ------------'
    dp.features.use_distance = True
    dp.read_data("portuguese")
    dp.train_perceptron(10)
    dp.test()
    '''
    Number of sentences: 3029
    Number of tokens: 25015
    Number of words: 7621
    Number of pos: 16
    Number of features: 46224
    Test accuracy (109 test instances): 0.714559386973
    '''

    print '------ use_contextual=TRUE ------------'
    dp.features.use_contextual = True
    dp.read_data("portuguese")
    dp.train_perceptron(10)
    dp.test()
    '''
    Number of sentences: 3029
    Number of tokens: 25015
    Number of words: 7621
    Number of pos: 16
    Number of features: 92918
    Test accuracy (109 test instances): 0.874521072797
    '''


# ---------------------------------- 4 ---------------------------------- #
'''
4. Which of the three important inference tasks discussed above (computing the most likely tree, computing the parti-
tion function, and computing the marginals) need to be performed in the structured perceptron algorithm?
R:

What about a maximum entropy classifier, with stochastic gradient descent?
R:

Check your answers by looking at the following two methods in code/dependency parser.py:
def train_perceptron(self, n_epochs):
...
def train_crf_sgd(self, n_epochs, sigma, eta0 = 0.001):
...

Repeat the last exercise by training a maximum entropy classifier, with stochastic gradient descent, using λ = 0.01
and a initial stepsize of η 0 = 0.1:
'''

if exerc == 4:
    print '#### Exerc. 4 #####'

    dp.train_crf_sgd(10, 0.01, 0.1)
    dp.test()

    '''
    Compare the results with those obtained by the perceptron algorithm.

    perceptron algorithm results:                                       crf_sgd results:

    Test accuracy (109 test instances): 0.495210727969          Test accuracy (109 test instances): 0.522988505747
    '''


# ---------------------------------- 5 ---------------------------------- #
'''
Train a parser for English using your favourite learning algorithm:
'''
if exerc == 5:
    print '#### Exerc. 5 #####'
    print '****** english *************'
    dp.read_data("english")
    dp.train_perceptron(10)
    dp.test()

    '''ENGLISH
    Number of sentences: 8044
    Number of tokens: 80504
    Number of words: 12202
    Number of pos: 48
    Number of features: 857
    Test accuracy (509 test instances): 0.451576786714
    '''


# ---------------------------------- 6 ---------------------------------- #
'''
Implement Eisner’s algorithm for projective dependency parsing. The pseudo-code is shown as Algo-
rithm 13. Implement this algorithm as the function
'''
if exerc == 6:
    print '#### Exerc. 6 #####'
    print '****** english *************'
    dp.features.use_lexical = True
    dp.features.use_distance = True
    dp.features.use_contextual = True
    dp.read_data("english")
    dp.projective = True
    dp.train_perceptron(10)
    dp.test()

    '''
    Test accuracy (509 test instances): 0.886732599366
    '''