__author__ = 'ruimendes'

import numpy as np
import Datasets.galton as galton
import matplotlib.pyplot as plt

GaltonData = galton.load()

print GaltonData

# What are the mean height and standard deviation of all the people in the sample? What is the mean height of the
# fathers and of the sons?


avgGlobal = GaltonData.mean()
stdGlobal = GaltonData.std()
avgFather = GaltonData[:, 0].mean()
avgSon = GaltonData[:, 1].mean()
print 'Mean height of all people: ' + str(avgGlobal)
print 'SD of all people: ' + str(stdGlobal)
print 'Mean height of all fathers: ' + str(avgFather)
print 'Mean height of all sons: ' + str(avgSon)

# Plot a histogram of all the heights (you might want to use the plt.hist function and the ravel method on
# arrays).
plt.hist(GaltonData.ravel())
plt.show()

# Plot the height of the father versus the height of the son.
plt.scatter(GaltonData[:, 0], GaltonData[:, 1])
plt.show()

print np.shape(GaltonData[:, 0])

print np.random.randn(2, 4)
fathers = GaltonData[:, 0] + np.random.randn(1, np.shape(GaltonData)[0])/10
sons = GaltonData[:, 1] + np.random.randn(1, np.shape(GaltonData)[0])/10
plt.scatter(fathers, sons)
plt.show()