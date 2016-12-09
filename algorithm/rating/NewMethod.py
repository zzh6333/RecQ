from baseclass.SocialRecommender import SocialRecommender
from data import rating
from tool import config
import numpy as np
import networkx as nx
import pickle




class NewMethod(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(NewMethod, self).__init__(conf,trainingSet,testSet,fold)
        self.config = conf

    def readConfiguration(self):
        super(NewMethod, self).readConfiguration()
        alpha = config.LineConfig(self.config['NewMethod'])
        eta = config.LineConfig(self.config['NewMethod'])
        self.alpha = float(alpha['-alpha'])
        self.eta = float(eta['-eta'])

    def printAlgorConfig(self):
        super(NewMethod, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'alpha: %.3f  eta: %.3f' % (self.alpha, self.eta)
        print '=' * 80

    def initModel(self):
        super(NewMethod, self).initModel()
        self.H = np.random.rand(self.dao.trainingSize()[0], self.k)
        #construct graph
        G = nx.DiGraph()
        for re in self.sao.relation:
            G.add_edge(re[0],re[1])
        #initialize new relation matrix
        self.S = self.sao.followees.copy()
        for u1 in self.S:
            for u2 in self.S[u1]:
                self.S[u1][u2] = np.random.rand()
        #compute betweenness
        self.getBetweenCentrality(G)

    def getBetweenCentrality(self,G,load=True):
        self.communication = np.zeros(len(self.sao.user))
        if not load:
            bt = nx.betweenness_centrality(G)
            output = open('between.pkl', 'wb')
            pickle.dump(bt, output)
        else:
            pkl_file = open('between.pkl', 'rb')
            bt = pickle.load(pkl_file)
        max = np.max(bt.values())
        min = np.min(bt.values())
        diff = max - min
        for u in bt:
            uid = self.dao.getUserId(u)
            self.communication[uid] = bt[u]-min/diff

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
                        if self.dao.rating(v,i)!=0:
                            vid = self.dao.getUserId(v)
                            id = self.dao.getItemId(i)
                            trustRating += self.S[u][v] * self.P[vid].dot(self.Q[id])
                            suv += self.S[u][v]

                u1 = u
                u = self.dao.getUserId(u)
                i = self.dao.getItemId(i)

                if suv!=0:
                    error1 = r - self.alpha*self.P[u].dot(self.Q[i])-(1-self.alpha)*(trustRating/suv)
                else:
                    error1 = r - self.P[u].dot(self.Q[i])

                error2 = self.communication[u] - self.P[u].dot(self.H[u])

                self.loss += error1**2 + self.eta*(error2**2)
                p = self.P[u].copy()
                q = self.Q[i].copy()

                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)

                #update latent vectors
                self.P[u] += self.lRate*(self.alpha*error1*q-self.regU*p + self.eta*error2*self.H[u])
                self.Q[i] += self.lRate*(self.alpha*error1*p-self.regI*q)
                self.H[u] += self.lRate*(self.eta*error2*self.P[u])
                if suv != 0:
                    for v in self.sao.getFollowees(u1):
                        vid = self.dao.getUserId(v)
                        self.S[u1][v] += self.lRate*((1-self.alpha)*error1*((self.P[u].dot(self.Q[i])*suv - trustRating)/(suv**2)))

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
