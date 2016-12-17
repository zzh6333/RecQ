from baseclass.SocialRecommender import SocialRecommender
from tool import config
import networkx as nx
from math import exp,sqrt
class gRSTE(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(gRSTE, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(gRSTE, self).readConfiguration()
        alpha = config.LineConfig(self.config['gRSTE'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(gRSTE, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: %.3f' %self.alpha
        print '='*80

    def initModel(self):
        super(gRSTE, self).initModel()
        G = nx.DiGraph()
        for re in self.sao.relation:
            G.add_edge(re[0], re[1])
        # pkl_file = open('between.pkl', 'rb')
        # bt = pickle.load(pkl_file)
        self.degree = nx.in_degree_centrality(G)
        minimum = min(self.degree.values()) + 1
        for key in self.degree:
            self.degree[key] = 1.0 / (1 + exp(-self.degree[key] / minimum)) + 1

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                u, i, r = entry
                error = r - self.predict(u,i)

                w = 1
                if self.degree.has_key(u) and self.degree[u] != 0:
                    w = self.degree[u]
                self.loss += w * error ** 2
                i = self.dao.getItemId(i)
                u = self.dao.getUserId(u)
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.loss += self.regU * p.dot(p) + self.regI * q.dot(q)
                # update latent vectors
                self.P[u] += self.lRate * (w*self.alpha*error * q - self.regU * p)
                self.Q[i] += self.lRate * (w*self.alpha*error * p - self.regI * q)
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
                if self.dao.containsUser(followee):  # followee is in rating set
                    fPred += weight * (self.P[uf].dot(self.Q[i]))
                    denom += weight
            u = self.dao.getUserId(u)
            if denom <> 0:
                return self.alpha * self.P[u].dot(self.Q[i])+(1-self.alpha)*fPred / denom
            else:
                return self.P[u].dot(self.Q[i])
        else:
            return self.dao.globalMean