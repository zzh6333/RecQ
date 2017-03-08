from baseclass.SocialRecommender import SocialRecommender
from math import exp,sqrt
import numpy as np
from tool import config
import networkx as nx
from math import log
#Social Recommendation Using Probabilistic Matrix Factorization
class gSoRec(SocialRecommender ):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(gSoRec, self).__init__(conf,trainingSet,testSet,relation,fold)


    def readConfiguration(self):
        super(gSoRec, self).readConfiguration()
        regZ = config.LineConfig(self.config['gSoRec'])
        self.regZ = float( regZ['-z'])

    def initModel(self):
        super(gSoRec, self).initModel()
        self.Z = np.random.rand(self.dao.trainingSize()[0], self.k)
        G = nx.DiGraph()
        for re in self.sao.relation:
            G.add_edge(re[0], re[1])
        # pkl_file = open('between.pkl', 'rb')
        # bt = pickle.load(pkl_file)
        self.degree = nx.in_degree_centrality(G)
        minimum = min(self.degree.values()) + 1
        for key in self.degree:
            self.degree[key] = 1.0 / (1 + exp(-self.degree[key] / minimum)) + 1
            # output = open('degree.pkl', 'wb')
            # pickle.dump(self.degree, output)
            # pkl_file = open('between.pkl', 'rb')
            # self.degree = pickle.load(pkl_file)

    def printAlgorConfig(self):
        super(gSoRec, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'regZ: %.3f' % self.regZ
        print '=' * 80


    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            #ratings
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u, i)
                w = 1
                if self.degree.has_key(u) and self.degree[u]!=0:
                    w = self.degree[u]
                i = self.dao.getItemId(i)
                u = self.dao.getUserId(u)
                self.loss += w*error ** 2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                # update latent vectors
                self.P[u] += self.lRate * (w*error * q - self.regU * p)
                self.Q[i] += self.lRate * (w*error * p - self.regI * q)

            #relations
            for entry in self.sao.relation:
                u, v, tuv = entry
                if self.dao.containsUser(u) and self.dao.containsUser(v):
                    vminus = len(self.sao.getFollowers(v))# ~ d - (k)
                    uplus = len(self.sao.getFollowees(u))#~ d + (i)
                    try:
                        weight = sqrt(vminus / (uplus + vminus + 0.0))
                    except ZeroDivisionError:
                        weight = 1

                    v = self.dao.getUserId(v)
                    u = self.dao.getUserId(u)
                    euv = weight * tuv - self.P[u].dot(self.Z[v])  # weight * tuv~ cik *
                    self.loss += self.regS * (euv ** 2)
                    p = self.P[u].copy()
                    z = self.Z[v].copy()
                    self.loss += self.regZ * z.dot(z)
                    # update latent vectors
                    self.P[u] += self.lRate * (self.regS * euv * z)
                    self.Z[v] += self.lRate * (self.regS * euv * p - self.regZ * z)
                else:
                    continue
            iteration += 1
            if self.isConverged(iteration):
                break
