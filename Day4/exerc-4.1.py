__author__ = 'ruimendes'


'''
In this simple exercise, you will see the CKY algorithm in action. There is a Javascript applet that illustrates
how CKY works (in its non-probabilistic form). Go to http://www.diotavelli.net/people/void/demos/
cky.html, and observe carefully the several steps taken by the algorithm. Write down a small grammar in CNF that
yields multiple parses for the ambiguous sentence The man saw the boy in the park with a telescope, and run the
demo for this particular sentence. What would happen in the probabilistic form of CKY?
'''

'''
S -> NP VP
NP -> DET N
NP -> NP PP
PP -> P NP
VP -> V NP

DET -> The
N -> man
V -> sees
DET -> the
N -> boy
P -> in
DET -> the
N -> park
P -> with
DET -> a
N -> telescope
'''

'''
In the probabilistic form of CKY we will have only one sparse tree (most probably) instead of we have several trees
'''