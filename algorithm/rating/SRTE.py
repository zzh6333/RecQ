from baseclass.SocialRecommender import SocialRecommender
from tool import qmath
import math
import numpy as np
from tool import config
from collections import defaultdict
from tool.qmath import denormalize
#Social Recommendation Using Text Embedding
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from structure.symmetricMatrix import SymmetricMatrix
from math import sqrt
import pickle
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


class SRTE(SocialRecommender ):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SRTE, self).__init__(conf,trainingSet,testSet,relation,fold)
        self.userSim = SymmetricMatrix(len(self.dao.user))
        self.trust = defaultdict(dict)

    def initModel(self):
        super(SRTE, self).initModel()
        self.r = defaultdict(dict)
        self.s = defaultdict(dict)
        for entry in self.dao.trainingData:
            userId, itemId, rating = entry

            # makes the rating within the range [0, 1].
            rating = denormalize(float(rating), self.dao.rScale[-1], self.dao.rScale[0])
            self.r[userId][itemId] = round(rating)

        re = Word_Eembedding_Method(self.r)
        for r in self.sao.relation:
            trustor,trustee,weight = r
            self.s[trustor][trustee] = weight
        se = Word_Eembedding_Method(self.s)




        # else:
        # pkl_file = open('dvecs.pkl', 'rb')
        # bt = pickle.load(pkl_file)
        self.uVecs = re.trainingNet(5,5,50)
        self.sVecs = se.trainingNet(5,3,50)
        output = open('dvecs.bin', 'wb')
        pickle.dump(self.uVecs, output)
        self.computeCorr()
        self.reComputeTrust()


    def readConfiguration(self):
        super(SRTE, self).readConfiguration()
        self.sim = self.config['similarity']
        self.shrinkage =int(self.config['num.shrinkage'])
        self.neighbors = int(self.config['num.neighbors'])

    def cosine_t(self,x1, x2):
        sum = x1.dot(x2)
        denom = sqrt(x1.dot(x1) * x2.dot(x2))
        try:
            return float(sum) / denom
        except ZeroDivisionError:
            return 0

    def computeCorr(self):
        'compute correlation among users'
        print 'Computing user correlation...'
        for u1 in self.dao.testSet_u:

            for u2 in self.dao.user:
                if u1 <> u2 and u1 in self.uVecs and u2 in self.uVecs:
                    if self.userSim.contains(u1, u2):
                        continue
                    sim = self.cosine_t(self.uVecs[u1], self.uVecs[u2])
                    self.userSim.set(u1, u2, sim)
            print 'user ' + u1 + ' finished.'
        print 'The user correlation has been figured out.'

    def reComputeTrust(self):
        print 're-computing user trust...'
        for u1 in self.dao.testSet_u :
            if u1 in self.sao.followees:
                for u2 in self.sao.followees[u1]:
                    if  u1 in self.sVecs and u2 in self.sVecs:
                        sim = self.cosine_t(self.sVecs[u1], self.sVecs[u2])
                        self.trust[u1][u2] = sim
                print 'user ' + u1 + ' finished.'
        print 'The user correlation has been figured out.'

    def predictBySim(self, u, i):
        if u not in self.uVecs:
            return self.dao.globalMean
        # find the closest neighbors of user u
        topUsers = sorted(self.userSim[u].iteritems(), key=lambda d: d[1], reverse=True)
        userCount = self.neighbors
        if userCount > len(topUsers):
            userCount = len(topUsers)
        # predict
        sum, denom = 0, 0
        for n in range(userCount):
            # if user n has rating on item i
            similarUser = topUsers[n][0]
            if self.dao.rating(similarUser, i) != 0:
                similarity = topUsers[n][1]
                rating = self.dao.rating(similarUser, i)
                sum += similarity * (rating - self.dao.userMeans[similarUser])
                denom += similarity
        if sum == 0:
            # no users have rating on item i,return the average rating of user u
            if not self.dao.containsUser(u):
                # user u has no ratings in the training set,return the global mean
                return self.dao.globalMean
            return self.dao.userMeans[u]
        pred = self.dao.userMeans[u] + sum / float(denom)
        return pred

    def predictByTrust(self,u,i):
        sum = 0
        denom = 0
        for t in self.trust[u]:
            if self.dao.contains(t,i):
                sum+=self.trust[u][t]*self.dao.rating(t,i)
                denom += self.trust[u][t]
        if sum!=0:
            return sum/denom
        else:
            return 0

    def predict(self,u,i):
        r1 = self.predictBySim(u,i)
        r2 = self.predictByTrust(u,i)
        if r2!=0:
            return (r1+r2)/2
        else:
            return r1

