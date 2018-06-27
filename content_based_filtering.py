import gensim
import os
import json
import csv
from functools import reduce
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

recUserId = "1"
tokenizer = RegexpTokenizer(r'\w+')

def getUserRatings(userId):
    result=[]
    with open('ratings-normal.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] == userId:
                result.append(row[1:])
    return result

def getMostPopularWords(dct, corpus, startIndex, endIndex):
    wordIndexes = list(map(lambda x: list(reversed(sorted(x,  key=lambda tup: tup[1])))[0][0], corpus[startIndex:endIndex]))
    return list(map(lambda x: dct[x], wordIndexes))

def getWords(plot):
    result = tokenizer.tokenize(plot)
    result = filter(lambda w: w not in stopwords.words('english'), result)
    return list(map(lambda w: w.lower(), result))

dirName = "data"
plotsDict = {}
titlesDict = {}
plots = []
for fname in os.listdir(dirName):
    fpath = os.path.join(dirName, fname)
    fhandle = open(fpath, 'r', encoding="utf8")
    contents = json.load(fhandle)
    currentIndex = int(fname[:-5])
    text = contents['Plot']
    plotsDict[currentIndex] = text
    titlesDict[currentIndex] = contents['Title']
    plots.append(text)

gen_docs = list(map(getWords, plots))

dictionary = gensim.corpora.Dictionary(gen_docs)

# tuples with the index of the word and the number of occurances
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.SparseMatrixSimilarity(
    tf_idf[corpus], num_features=len(dictionary))

userRatings = getUserRatings(recUserId)
goodUserRatings = list(filter(lambda ur: float(ur[1]) > 4, userRatings))
goodUserMovieIndexes = list(map(lambda ur: int(ur[0]), goodUserRatings))
queryStr = reduce(lambda accum, curr: accum + " " + plotsDict[curr], goodUserMovieIndexes, "")
query_doc = getWords(queryStr)
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]

sim_matrix = sims[query_doc_tf_idf]
sortedM = list(reversed(sorted(range(len(sim_matrix)), key=lambda i: sim_matrix[i])))
top5 = []
for ind in sortedM:
    movieIndex = ind + 1
    if movieIndex not in goodUserMovieIndexes:
        top5.append(movieIndex)
    if len(top5) == 5:
        break

likedMovieTitles = list(map(lambda m: titlesDict[m], goodUserMovieIndexes))
recMovieTitles = list(map(lambda m: titlesDict[m], top5))

print(top5)
print("User likes \n%s\n\n" % "\n".join(likedMovieTitles))
print("Recommended \n%s\n\n" % "\n".join(recMovieTitles))
