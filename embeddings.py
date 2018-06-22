import gensim
import os
import json
from nltk.tokenize import word_tokenize

# class SentencesCollector:
#     def __init__(self, dirname):
#         self.dirname = dirname

#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             fpath = os.path.join(self.dirname, fname)
#             fhandle = open(fpath, 'r')
#             contents = json.load(fhandle)
#             words = contents['Plot'].split()
#             yield words

# sentences = SentencesCollector("data")
# model = gensim.models.Word2Vec(sentences, min_count=1)
# print("vector of the word 'deadly'" % model['deadly'])
# print("model.similarity('game', 'battle') is %s" % model.similarity('game', 'battle'))

dirName = "data"
plots = []
for fname in os.listdir(dirName):
    fpath = os.path.join(dirName, fname)
    fhandle = open(fpath, 'r', encoding="utf8")
    contents = json.load(fhandle)
    text = contents['Plot']
    plots.append(text)

gen_docs = [[w.lower() for w in word_tokenize(plot)] for plot in plots]

dictionary = gensim.corpora.Dictionary(gen_docs)

# tuples with the index of the word and the number of occurances
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.SparseMatrixSimilarity(tf_idf[corpus], num_features=len(dictionary))

query_doc = [w.lower() for w in word_tokenize("ship iceberg drama titanic love")]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]

a = sims[query_doc_tf_idf]
print(a)