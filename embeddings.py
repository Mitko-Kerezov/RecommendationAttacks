import gensim
import os
import json

class SentencesCollector:
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fpath = os.path.join(self.dirname, fname)
            fhandle = open(fpath, 'r')
            contents = json.load(fhandle)
            words = contents['Plot'].split()
            yield words

sentences = SentencesCollector("data")
model = gensim.models.Word2Vec(sentences, min_count=1)
print("vector of the word 'deadly'" % model['deadly'])
print("model.similarity('game', 'battle') is %s" % model.similarity('game', 'battle'))
