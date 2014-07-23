__author__ = 'ruimendes'

import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.naive_bayes as nb
scr = srs.SentimentCorpus("books")

from lxmls.classifiers.multinomial_naive_bayes import *



