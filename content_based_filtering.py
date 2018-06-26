import gensim
import os
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

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

tokenizer = RegexpTokenizer(r'\w+')
def getWords(plot):
    result = tokenizer.tokenize(plot)
    result = filter(lambda w: w not in stopwords.words('english'), result)
    return list(map(lambda w: w.lower(), result))

dirName = "data"
queryStr = ""
plots = []
for fname in os.listdir(dirName):
    fpath = os.path.join(dirName, fname)
    fhandle = open(fpath, 'r', encoding="utf8")
    contents = json.load(fhandle)
    text = contents['Plot']
    if fname in ["0" + str(ind) + ".json" for ind in range(61, 66)]:
        queryStr += text
    plots.append(text)

gen_docs = list(map(getWords, plots))

dictionary = gensim.corpora.Dictionary(gen_docs)

# tuples with the index of the word and the number of occurances
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.SparseMatrixSimilarity(
    tf_idf[corpus], num_features=len(dictionary))

print(queryStr)
query_doc = getWords(queryStr)
query_doc_bow = dictionary.doc2bow(query_doc)
# query_doc_bow = [(2158, 5), (2194, 1), (2196, 3), (2228, 2), (2198, 2)]
query_doc_tf_idf = tf_idf[query_doc_bow]

sim_matrix = sims[query_doc_tf_idf]
print(sim_matrix)
sortedM = list(reversed(sorted(range(len(sim_matrix)), key=lambda i: sim_matrix[i])))
print(sortedM)
