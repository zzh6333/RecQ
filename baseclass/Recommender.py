# Copyright (C) 2016 School of Software Engineering, Chongqing University
#
# This file is part of RecQ.
#
# RecQ is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from data.rating import RatingDAO
from tool.file import FileIO
from tool.qmath import denormalize
from tool.config import Config,LineConfig
from os.path import abspath
from time import strftime,localtime,time
from evaluation.measure import Measure
class Recommender(object):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        self.config = conf
        self.dao = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.dao = RatingDAO(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.measure = []

    def readConfiguration(self):
        self.algorName = self.config['recommender']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        print 'Algorithm:',self.config['recommender']
        print 'Ratings dataset:',abspath(self.config['ratings'])
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print 'Test set:',abspath(LineConfig(self.config['evaluation.setup']).getOption('-testSet'))
        #print 'Count of the users in training set: ',len()
        print 'Training set size: (user count: %d, item count %d, record count: %d)' %(self.dao.trainingSize())
        print 'Test set size: (user count: %d, item count %d, record count: %d)' %(self.dao.testSize())
        print '='*80

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self,u,i):
        pass

    def checkRatingBoundary(self,prediction):
        if prediction > self.dao.rScale[-1]:
            return self.dao.rScale[-1]
        elif prediction < self.dao.rScale[0]:
            return self.dao.rScale[0]
        else:
            return round(prediction,3)

    def evalRatings(self):
        res = [] #used to contain the text of the result
        res.append('userId  itemId  original  prediction\n')
        #predict
        for ind,entry in enumerate(self.dao.testData):
            user,item,rating = entry

            #predict
            prediction = self.predict(user,item)
            #denormalize
            prediction = denormalize(prediction,self.dao.rScale[-1],self.dao.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.dao.testData[ind].append(pred)
            res.append(user+' '+item+' '+str(rating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        #output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print 'The Result has been output to ',abspath(outDir),'.'
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.dao.testData)
        FileIO.writeFile(outDir, fileName, self.measure)

    def evalRanking(self):
        res = []  # used to contain the text of the result
        N = int(self.ranking['-topN'])
        if N>100 or N<0:
            N=100
        res.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is matched\n')
        # predict
        topNSet = {}
        userCount = len(self.dao.testSet_u)
        for i,user in enumerate(self.dao.testSet_u):
            itemSet = []
            line = user+':'

            for item in self.dao.item:
                if not self.dao.rating(user,item):
                    # predict
                    prediction = self.predict(user, item)
                    # denormalize

                    prediction = denormalize(prediction, self.dao.rScale[-1], self.dao.rScale[0])

                    prediction = round(prediction,4)
                    #pred = self.checkRatingBoundary(prediction)
                    #####################################
                    # add prediction in order to measure
                    itemSet.append((item,prediction))

            itemSet.sort(key=lambda d:d[1],reverse=True)
            topNSet[user] = itemSet[0:N]

            if i%100==0:
                print self.algorName,self.foldInfo,'progress:'+str(i)+'/'+str(userCount)
            for item in topNSet[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if self.dao.testSet_u[user].has_key(item[0]):
                    line+='*'

            line+='\n'
            res.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender'] + '@' + currentTime + '-top-'+str(N)+'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print 'The Result has been output to ', abspath(outDir), '.'
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.dao.testSet_u,topNSet,N)
        FileIO.writeFile(outDir, fileName, self.measure)

    def execute(self):
        self.readConfiguration()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print 'Loading model %s...' %(self.foldInfo)
            self.loadModel()
        else:
            print 'Initializing model %s...' %(self.foldInfo)
            self.initModel()
            print 'Building Model %s...' %(self.foldInfo)
            self.buildModel()

        #preict the ratings or item ranking
        print 'Predicting %s...' %(self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()

        #save model
        if self.isSaveModel:
            print 'Saving model %s...' %(self.foldInfo)
            self.saveModel()

        return self.measure


