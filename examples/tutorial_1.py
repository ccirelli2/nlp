"""
Beginners Tutorial(s)
References:
    https://www.machinelearningplus.com/nlp/gensim-tutorial/

Input:
    Gensim requires the words / tokens to be converted to unique ids,
    which can be input into Gensim as dictionary objects.

"""
###############################################################################
# Import Libraries
###############################################################################
import os
import sys
import gensim
import numpy as np

from pprint import pprint
from decouple import config as d_config
from gensim import corpora, models
from gensim.utils import simple_preprocess
from smart_open import smart_open



###############################################################################
# Directories
###############################################################################
DIR_ROOT = d_config('DIR_ROOT')
DIR_SRC = os.path.join(DIR_ROOT, 'src')
DIR_DATA = os.path.join(DIR_ROOT, 'data')
DIR_RESULTS = os.path.join(DIR_ROOT, 'results')
[sys.path.append(x) for x in [DIR_ROOT, DIR_SRC]]


###############################################################################
# Project Modules
###############################################################################
import utility as my_utils


###############################################################################
# How to create a dictionary from a list of scentences?
###############################################################################
documents = ["The Saudis are preparing a report that will acknowledge that",
             "Saudi journalist Jamal Khashoggi's death was the result of an",
             "interrogation that went wrong, one that was intended to lead",
             "to his abduction from Turkey, according to two sources."]

documents_2 = ["One source says the report will likely conclude that",
                "the operation was carried out without clearance and",
                "transparency and that those involved will be held",
                "responsible. One of the sources acknowledged that the",
                "report is still being prepared and cautioned that",
                "things could change."]

# Step1: Tokenize words
text1 = [[text for text in doc.split()] for doc in documents]

# Step2: Create Dictionary
dict1 = corpora.Dictionary(text1)

# Attributes
dict_token_cnts = dict1.token2id
token_frequency = dict1.cfs
doc_frequency = dict1.dfs
num_docs = dict1.num_docs
num_pos = dict1.num_pos

# Methods
dict1_most_freq_toks = dict1.most_common()


# Add to Existing Dictionary
text2 = [[text for text in doc.split()] for doc in documents_2]
dict1.add_documents(text2)

# Verify by getting the number od couments.  Should == number of sentences
num_docs = dict1.num_docs
assert num_docs == len(documents) + len(documents_2)


###############################################################################
# Create Dictionary From Text Files 
###############################################################################
'''
    Note: Gensim will read the file in line by line as opposed to loading into
    memory the entire file before pre-processing / constructing the
    dictionary.

    simple_preporcess: convert a document into a list of tokens,
        ignoring tokens that are too short or too long.
        ::min_len:
        ::max_len:
'''
dict3 = corpora.Dictionary(simple_preprocess(line, deacc=True)
    for line in open(os.path.join(DIR_DATA, 'sample.txt'),
    encoding='utf-8'))


###############################################################################
# Create Dictionary From Many Text Files 
###############################################################################
'''Requires that we create our own class object that iterates through the
    files + lines.
'''
class ReadTxtFiles(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='latin'):
                yield simple_preprocess(line)

dict4 = corpora.Dictionary(ReadTxtFiles(DIR_DATA))


###############################################################################
# Create Bag-Of-Words Corpus 
###############################################################################
'''Gensims' equivalent of Document-Term-Matrix
'''
# Documents (Each List of Text is a document)
my_docs = ["Who let the dogs out?",
           "Who? Who? Who? Who?"]

# Tokenize Each Document 
tokenized_list = [simple_preprocess(doc) for doc in my_docs]

# Create Corpus
my_dict = corpora.Dictionary()
my_corpus = [my_dict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
'''
>>> print(my_corpus)
>>> [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], [(4, 4)]]

[[]] the frequencies are seperated into list objects.
(,) the 0 position of the tuple is the id and the 1 position is the token
    frequency.  The position in the nested list refers to the document.

(0,1) means that word with id-0 appears once in the first document.
(4,4) means the word with the id 4, which is "Who" appears 4 times in the
    second document.
use my_dict.token2id to get the tokens and their ids.
'''
# Converte IDs to Words
word_counts = [[(my_dict[id], count) for id, count in doc] for doc in my_corpus]
assign_num2docs = list(enumerate(word_counts))


###############################################################################
# How to save a gensim dictionary and corpus to disk and load them back 
###############################################################################
'tbd'

###############################################################################
# Create TFIDF Matrix in Gensim 
###############################################################################
# Define docs
tfidf_docs = [
    'This is the first document. This shit',
    'This is the second document. This balls',
    'This is the third document. This crap']

# Create a Dictionary & Corpus
my_dict = corpora.Dictionary([simple_preprocess(line) for line in tfidf_docs])
my_corpus = [my_dict.doc2bow(simple_preprocess(line)) for line in tfidf_docs]

# Create TF-IDF Model
tfidf_model = models.TfidfModel(my_corpus, smartirs='ntc')

# Expose Word Frequencies
'''
for doc in my_corpus:
    print([[my_dict[id], freq] for id, freq in doc])

# Expose Word TFIDF Weights
for doc in tfidf_model[my_corpus]:
    print([[my_dict[id], np.around(freq, decimals=2)] for id, freq in doc])
'''


















































 


