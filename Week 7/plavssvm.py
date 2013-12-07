# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:36:34 2013

@author: dyanna
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

def getSample(pointA, pointB, numberOfPoints):
    pointList = list(zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints)))
    sample = np.array([(i[0], i[1], isLeft(pointA, pointB, i)) for i in pointList])
    y = sample[:,2]
    breakpoint = False
    while not breakpoint:
        if(len(y[y==-1]) == 0 or len(y[y==1]) == 0):
            pointList = list(zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints)))
            sample = np.array([(i[0], i[1], isLeft(pointA, pointB, i)) for i in pointList])
            y = sample[:,2]
        else: 
            breakpoint = True
    return sample

def getRandomLine():
    return list(zip(np.random.uniform(-1,1.00,2),np.random.uniform(-1,1.00,2)))


def isLeft(a, b, c):
    return 1 if ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0 else -1;

def sign(x):
    return 1 if x > 0 else -1 


def doPLA(sample):
    w = np.array([0,0,0])
    iteration = 0
    it = 0
    while True:#(it < 10):
        iteration = iteration + 1
        it = it + 1
        mismatch = list()
        for i in sample:
            #print("point in question ", i , " weight ", w)
            yy = w[0] + w[1] * i[0] + w[2] * i[1]
            #print("this is after applying weight to a point ",yy)
            point = [i[0], i[1], sign(yy)]
            if any(np.equal(sample, point).all(1)):
                #print "point not in sample"
                if(point[2] == -1):
                    mismatch.append((1, (i[0]), (i[1])))
                else:
                    mismatch.append((-1, -(i[0]), -(i[1])))
        #print " length ", len(mismatch), " mismatch list ",mismatch 
        if(len(mismatch) > 0):
            #find a random point and update w
            choiceIndex = np.random.randint(0, len(mismatch))
            choice = mismatch[choiceIndex]
            #print("choice ", choice)
            w = w + choice
            #print "new weight ", w
        else:
            break
    #print("this is the iteration ", iteration)
    #print("this is the weight ", w)
    #montelist = [monetcarlo((x1,y1),(x2,y2),w,10000) for i in range(5)]
    #print("Montelist " , montelist)
    #monteavg = sum([i for i in montelist])/10
    return w, iteration

def getMisMatches(data, weights):
    #print data
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]
    results = -1 * results
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)

def doMonteCarloQP(pointa, pointb, clf, pclf, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    #print points
    dataset_Monte = np.array([(i[0],i[1], isLeft(pointa,pointb,i)) for i in points])
    X = np.c_[dataset_Monte[:,0], dataset_Monte[:,1]]
    y = dataset_Monte[:,2]
    #print dataset_Monte
    return 1-pclf.score(X,y), 1-clf.score(X,y)
    

if __name__ == "__main__":
    clf = SVC(C = 1000, kernel = 'linear')  
    pclf = Perceptron()
    avgpla = list()
    avglf = list()
    perc = 0
    for k in range(1000):
        nopoints = 100
        line = getRandomLine()
        sample = getSample(line[0], line[1], nopoints)
            #print(sample)
        X = np.c_[sample[:,0], sample[:,1]]
        y = sample[:,2]
        #print(y)
        pclf.fit(X,y)
        clf.fit(X,y)
        #print(clf.score(X,y))
        #w, it = doPLA(sample)
        pscore, svmscore = doMonteCarloQP(line[0], line[1], clf, pclf, 50000)
        avgpla.append(pscore)
        avglf.append(svmscore)
        if(svmscore < pscore):
            perc = perc + 1
    print(sum(avgpla)/len(avgpla))
    print(sum(avglf)/len(avglf))
    print("%age ", perc/len(avgpla))
        

    