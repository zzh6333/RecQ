from baseclass.SocialRecommender import SocialRecommender
from structure.symmetricMatrix import SymmetricMatrix
from tool import config,qmath
import numpy as np
import networkx as nx
import pickle




class DTrustMF(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DTrustMF, self).__init__(conf,trainingSet,testSet,fold)
        self.config = conf

    def readConfiguration(self):
        super(DTrustMF, self).readConfiguration()
        alpha = config.LineConfig(self.config['DTrustMF'])
        eta = config.LineConfig(self.config['DTrustMF'])
        self.alpha = float(alpha['-alpha'])
        self.eta = float(eta['-eta'])

    def printAlgorConfig(self):
        super(DTrustMF, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'alpha: %.3f' % self.alpha
        print '=' * 80

    def initModel(self):
        super(DTrustMF, self).initModel()
        self.Sim = SymmetricMatrix(len(self.dao.user))
        for user in self.dao.user:
            followees = self.sao.getFollowees(user)
            if len(followees):
                for followee in followees:
                    if self.dao.containsUser(followee):
                        rowUser = self.dao.row(user)
                        rowFollowee = self.dao.row(followee)
                        self.Sim.set(user, followee, qmath.cosine(rowUser, rowFollowee))

        self.S = self.sao.followees.copy()

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for triple in self.dao.trainingData:
                u,i,r = triple
                trustRating = 0
                suv = 0
                if len(self.sao.getFollowees(u)) != 0:
                    for v in self.sao.getFollowees(u):
                        if self.dao.containsUser(v):
                            trustRating += self.S[u][v] * self.dao.rating(v,i)
                            suv += self.S[u][v]
                uid = self.dao.getUserId(u)
                iid = self.dao.getItemId(i)
                if suv!=0:
                    error = r - self.alpha*self.P[uid].dot(self.Q[iid])-(1-self.alpha)*(trustRating/suv)
                else:
                    error = r - self.P[uid].dot(self.Q[iid])
                self.loss += error**2
                p = self.P[uid].copy()
                q = self.Q[iid].copy()

                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)

                #update latent vectors
                self.P[uid] += self.lRate*(self.alpha*error*q-self.regU*p)
                self.Q[iid] += self.lRate*(self.alpha*error*p-self.regI*q)
                if suv != 0:
                    for v in self.sao.getFollowees(u):
                        if self.dao.containsUser(v):
                            self.S[u][v] += self.lRate*((1-self.alpha)*error*((self.dao.rating(v,i)*suv - trustRating)/(suv**2)))
            iteration += 1
            if self.isConverged(iteration):
                break

        self.sao.followees = self.S



    def predict(self,u,i):
        # if self.dao.containsUser(u) and self.dao.containsItem(i):
        #     u = self.dao.getUserId(u)
        #     i = self.dao.getItemId(i)
        #
        #     if len(self.sao.getFollowees(u)) != 0:
        #         self.cTrustRating = 0
        #         for v in self.sao.getFollowees(u):
        #             self.cTrustRating += self.S[u][v] * self.communication[u][v]*self.dao.rating(v, i)
        #     else:
        #         pass
        #     return self.alpha*self.P[u].dot(self.Q[i])+(1-self.alpha)*(self.cTrustRating/suv)
        # else:
        #     return self.dao.globalMean
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            i = self.dao.getItemId(i)
            fPred = 0
            denom = 0
            relations = self.sao.getFollowees(u)
            for followee in relations:
                weight = relations[followee]
                uf = self.dao.getUserId(followee)
                if uf <> -1 and self.dao.containsUser(followee):  # followee is in rating set
                    fPred += weight * (self.P[uf].dot(self.Q[i]))
                    denom += weight
            u = self.dao.getUserId(u)
            if denom <> 0:
                return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
            else:
                return self.P[u].dot(self.Q[i])
        else:
            return self.dao.globalMean
