from baseclass.SocialRecommender import SocialRecommender
import numpy as np
import networkx as nx
import pickle
class SocialMF(SocialRecommender ):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SocialMF, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(SocialMF, self).readConfiguration()

    def initModel(self):
        super(SocialMF, self).initModel()
        self.H = np.random.rand(self.k, self.k)
        G = nx.DiGraph()
        for re in self.sao.relation:
            G.add_edge(re[0], re[1])
        # pkl_file = open('between.pkl', 'rb')
        # bt = pickle.load(pkl_file)
        self.degree = nx.in_degree_centrality(G)
        output = open('between.pkl', 'wb')
        pickle.dump(self.degree, output)
        # pkl_file = open('between.pkl', 'rb')
        # self.degree = pickle.load(pkl_file)


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                userId, itemId, r = entry
                followees = self.sao.getFollowers(userId)
                u = self.dao.getUserId(userId)
                i = self.dao.getItemId(itemId)
                error = r - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                fPred = 0
                denom = 0
                relationLoss = np.zeros(self.k)
                for followee in followees:
                    weight= followees[followee]
                    uf = self.dao.getUserId(followee)
                    if uf <> -1 and self.dao.containsUser(followee):
                        fPred += weight * self.P[uf]
                        denom += weight
                if denom <> 0:
                    relationLoss = p - fPred / denom

                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q) + self.regS *  relationLoss.dot(relationLoss)

                # update latent vectors
                self.P[u] += self.lRate * (error * q - self.regU * p - self.regS * relationLoss)
                self.Q[i] += self.lRate * (error * p - self.regI * q)


            iteration += 1
            if self.isConverged(iteration):
                break
