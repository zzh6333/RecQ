from baseclass.IterativeRecommender import IterativeRecommender
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from collections import defaultdict
from tool.qmath import denormalize,normalize
from tool.qmath import l2
import numpy as np
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if key not in flipped:
                flipped[key] = [value]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            yield LabeledSentence(source, [prefix])

    def to_array(self):
        self.sentences = []
        for prefix, source in self.sources.items():
            self.sentences.append(LabeledSentence(source, [prefix]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


class Word_Eembedding_Method(object):
    def __init__(self, trainingData):
        self.corpus = {}
        self.trainingDict = trainingData

        for user in trainingData:
            tag = user
            self.corpus[tag] = []
            for item, rating in trainingData[user].iteritems():
                self.corpus[tag] += int(rating)*[item]


    def trainingNet(self, epoch,window, nDimension):
        self.nDimension = nDimension
        sentences = LabeledLineSentence(self.corpus)
        self.model = Doc2Vec(min_count=1, window=window, size=nDimension, sample=1e-4, negative=5, workers=4)
        corpus = sentences.to_array()
        self.model.build_vocab(corpus)
        for epoch in range(epoch):
            self.model.train(sentences.sentences_perm())
        return self.model.docvecs


class W2V2(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(W2V2, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(W2V2, self).initModel()
        self.userProfile = defaultdict(dict)
        self.itemProfile = defaultdict(dict)
        self.Bu = np.random.rand(self.dao.trainingSize()[0]) / 10  # biased value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1]) / 10  # biased value of item
        self.H = np.random.rand(50, 50) / 10
        for entry in self.dao.trainingData:
            userId, itemId, rating = entry

            # makes the rating within the range [0, 1].
            rating = denormalize(float(rating), self.dao.rScale[-1], self.dao.rScale[0])
            self.userProfile[userId][itemId] = round(rating)
            self.itemProfile[itemId][userId] = round(rating)

        up = Word_Eembedding_Method(self.userProfile)

        ip = Word_Eembedding_Method(self.itemProfile)


        self.uVecs = up.trainingNet(5, 5, 50)
        self.iVecs = ip.trainingNet(5, 5, 50)


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u, i)
                x = np.array([self.uVecs[u]/10])
                y = np.array([self.iVecs[i]/10])
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)
                self.loss += error ** 2

                bu = self.Bu[u]
                bi = self.Bi[i]
                self.loss += self.regB * bu ** 2 + self.regB * bi ** 2+self.regB*l2(self.H)

                # update latent vectors
                self.Bu[u] += self.lRate * (error - self.regB * bu)
                self.Bi[i] += self.lRate * (error - self.regB * bi)
                self.H += self.lRate * (error*(x-y).T.dot(x-y))
            iteration += 1
            if self.isConverged(iteration):
                break

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            x = np.array([self.uVecs[u]/10])
            y = np.array([self.iVecs[i]/10])
            u = self.dao.getUserId(u)
            i = self.dao.getItemId(i)
            return self.dao.globalMean + self.Bi[i] + self.Bu[u] + (x-y).dot(self.H).dot((x-y).T)[0][0]
        else:
            return self.dao.globalMean