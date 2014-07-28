__author__ = 'ruimendes'


'''
Using the Word Count example we’ve given you above, implement the Na ̈ıve Bayes language detector de-
scribed above. You should do this in two parts:
• Steps 1 to 3 (counting occurrences of trimers in train data, for both languages), should be run on EC2.
• Step 4 should be run on your AWS server (it is quite fast even with just one computer). It should imple-
ment the formula given in equation
'''

# wordcount.py
'''
Code to count the number of words per document.
It each node in the cluster will call the method in this code.
The Map method counts the total number of words per document.
the Reduce method aggregates the sum of the words for all documents.
'''

# trimercount.py
'''
Similar to the above, but instead of counting words, it counts the total occurrences of 3-block characters.
 Example: 'I love football' => 'I l', 'ove', ' fo', 'otb', 'all'
'''

# postprocess
'''
It detects the language of a given sentence based on Nayve Bayes Algorithm use of trimmers.
It starts by processing the sentence in trimmers.
For each trimmer, it checks in the respective dictionary (en.counts.txt or pt.counts.txt) if it is in it.
If it is, it returns the value count), otherwise it returns the value 1 (example: tri_en = counts_en.get(tri, 1.0))
Then it calculates the log ratio for each word belonging to each language.
(Example: log_prob_tri_pt = math.log10(tri_pt/total_trimers_pt) and log_prob_tri_en = math.log10(tri_en/total_trimers_en))
Finally, returns the result regarding the difference among the log probabilities of each language.
 Note: if the difference is big enough (abs(val) >= 5) then the system is sure.
 if abs(val) >= 5:
        print "This is a", language, "sentence."
    else:
        print "This seems to be a", language, "sentence, but I'm not sure."
'''