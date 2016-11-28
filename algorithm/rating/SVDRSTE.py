from baseclass.SocialRecommender import SocialRecommender
from tool import config
import numpy as np
class SVDRSTE(SocialRecommender):
    def __init__(self,conf):
        super(SVDRSTE, self).__init__(conf)

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        alpha = config.LineConfig(self.config['RSTE'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(SVDRSTE, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(SVDRSTE, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])  # biased value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])  # biased value of item

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u,i)
                i = self.dao.getItemId(i)
                u = self.dao.getUserId(u)
                self.loss += error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                bu = self.Bu[u]
                bi = self.Bi[i]
                self.loss += self.regB*bu**2 + self.regB*bi**2
                #update latent vectors
                self.Bu[u] = bu+self.lRate*(error-self.regB*bu)
                self.Bi[i] = bi+self.lRate*(error-self.regB*bi)
                # update latent vectors
                self.P[u] = p + self.lRate * (self.alpha*error * q - self.regU * p)
                self.Q[i] = q + self.lRate * (self.alpha*error * p - self.regI * q)
            iteration += 1
            self.isConverged(iteration)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            i = self.dao.getItemId(i)
            fPred = 0
            denom = 0
            relations = self.sao.getFollowees(u)
            for followee in relations:
                weight = relations[followee]
                uf = self.dao.getUserId(followee)
                if uf <> -1 and self.dao.containsUser(uf):  # followee is in rating set
                    fPred += weight * (self.P[uf].dot(self.Q[i]))
                    denom += weight
            u = self.dao.getUserId(u)
            if denom <> 0:
                return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom +self.dao.globalMean+self.Bi[i]+self.Bu[u]
            else:
                return self.P[u].dot(self.Q[i])+self.dao.globalMean+self.Bi[i]+self.Bu[u]
        else:
            return self.dao.globalMean