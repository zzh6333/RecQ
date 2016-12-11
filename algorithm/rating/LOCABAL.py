from baseclass.SocialRecommender import SocialRecommender
from tool import config
import numpy as np
import networkx as nx
import math
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix

class LOCABAL(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(LOCABAL, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(LOCABAL, self).readConfiguration()
        alpha = config.LineConfig(self.config['LOCABAL'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(LOCABAL, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(LOCABAL, self).initModel()
        self.H = np.random.rand(self.k,self.k)
        G = nx.DiGraph()
        for re in self.sao.relation:
            G.add_edge(re[0], re[1])
        pr = nx.pagerank(G, alpha=0.85)
        self.W = {}
        for u in pr:
            self.W[u] = 1/(1+math.exp(pr[u]))
        self.S = SymmetricMatrix(len(self.dao.user))
        for user in self.dao.user:
            followees = self.sao.getFollowees(user)
            if len(followees):
                for followee in followees:
                    if self.dao.containsUser(followee):
                        rowUser = self.dao.row(user)
                        rowFollowee = self.dao.row(followee)
                        self.S.set(user,followee,qmath.cosine(rowUser,rowFollowee))




    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                weight = 0
                if self.dao.rating(u,i):
                    if self.W.has_key(u):
                        weight = math.sqrt(self.W[u])
                error = r - self.predict(u,i)
                id = self.dao.getItemId(i)
                uid = self.dao.getUserId(u)
                self.loss += error ** 2
                p = self.P[uid].copy()
                q = self.Q[id].copy()
                H1 = self.H.copy()
                self.loss += weight*error**2+ self.regU * p.dot(p) + self.regI * q.dot(q) + qmath.l2(self.H)
                deltaH = np.zeros((self.k,self.k))

                for followee in self.sao.getFollowees(u):
                    if self.S.contains(u,followee) and self.dao.containsUser(followee):
                        k = self.dao.getUserId(followee)
                        localLoss = self.S[u][followee] - np.dot(np.dot(p.T,H1),self.P[k])

                        self.loss += localLoss ** 2
                        deltaH += (self.S[u][followee] - np.dot(np.dot(p.T,H1),self.P[k]))* \
                                  np.dot(p.reshape(self.k,1),self.P[k].reshape(1,self.k))

                # update latent vectors
                self.P[uid] += self.lRate * (weight*error * q - self.regU * p)
                self.Q[id] += self.lRate * (weight*error * p - self.regI * q)
                self.H += self.lRate * (self.alpha * deltaH)

            iteration += 1
            self.isConverged(iteration)

    # def predict(self,u,i):
    #     if self.dao.containsUser(u) and self.dao.containsItem(i):
    #         i = self.dao.getItemId(i)
    #         fPred = 0
    #         denom = 0
    #         relations = self.sao.getFollowees(u)
    #         for followee in relations:
    #             weight = relations[followee]
    #             uf = self.dao.getUserId(followee)
    #             if uf <> -1 and self.dao.containsUser(followee):  # followee is in rating set
    #                 fPred += weight * (self.P[uf].dot(self.Q[i]))
    #                 denom += weight
    #         u = self.dao.getUserId(u)
    #         if denom <> 0:
    #             return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
    #         else:
    #             return self.P[u].dot(self.Q[i])
    #     else:
    #         return self.dao.globalMean