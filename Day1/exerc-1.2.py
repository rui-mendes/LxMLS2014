__author__ = 'ruimendes'

import matplotlib.pyplot as plt
import lxmls.readers.simple_data_set as sds
import lxmls.readers.sentiment_reader as srs
scr = srs.SentimentCorpus("books")

sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1], balance=0.5, split=[0.5, 0, 0.5])

import lxmls.classifiers.perceptron as percc
#
# perc = percc.Perceptron()
# params_perc_sd = perc.train(sd.train_X, sd.train_y)
# y_pred_train = perc.test(sd.train_X, params_perc_sd)
# acc_train = perc.evaluate(sd.train_y, y_pred_train)
# y_pred_test = perc.test(sd.test_X, params_perc_sd)
# acc_test = perc.evaluate(sd.test_y, y_pred_test)
#
# print "Perceptron Simple Dataset Accuracy train: %f test: %f" % (acc_train, acc_test)
#
# fig, axis = sd.plot_data()
# fig, axis = sd.add_line(fig, axis, params_perc_sd, "Perceptron", "blue")
#
# # Plot the decision boundary found
# plt.show(sd)



#################### TODO ##################################
# Run the perceptron algorithm on the Amazon dataset.
perc = percc.Perceptron()
params_perc_scr = perc.train(scr.train_X, scr.train_y)
y_pred_train = perc.test(scr.train_X, params_perc_scr)
acc_train = perc.evaluate(scr.train_y, y_pred_train)
y_pred_test = perc.test(scr.test_X, params_perc_scr)
acc_test = perc.evaluate(scr.test_y, y_pred_test)

print "Perceptron Amaon Dataset Accuracy train: %f test: %f" % (acc_train, acc_test)

# fig, axis = sd.plot_data()
# fig, axis = sd.add_line(fig, axis, params_perc_sd, "Perceptron", "blue")
#
# # Plot the decision boundary found
plt.show(scr)