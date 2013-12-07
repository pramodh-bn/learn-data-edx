# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:55:28 2013

@author: dyanna
"""

import numpy as np
from sklearn.svm import SVC

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

def getPoints(numberOfPoints):
    pointList = list(zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints)))
    return pointList

def isLeft(a, b, c):
    return 1 if ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0 else -1;

def sign(x):
    return 1 if x > 0 else -1 


def getMisMatchesQP(data, clf):
    #print(data)
    data_x = np.c_[data[:,0], data[:,1]]
    results = clf.predict(data_x)
    #print(np.sign(results))
    print("mismatch ", float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data))
    print("score ", clf.score(data_x, data[:,2]))
    
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)
    


def doMonteCarloQP(pointa, pointb, clf, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    #print points
    dataset_Monte = np.array([(i[0],i[1], isLeft(pointa,pointb,i)) for i in points])
    #print dataset_Monte
    return getMisMatchesQP(dataset_Monte, clf)

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
    


def doMonteCarloNP(pointa, pointb, weights, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    #print points
    dataset_Monte = np.array([(i[0],i[1], isLeft(pointa,pointb,i)) for i in points])
    #print dataset_Monte
    return getMisMatches(dataset_Monte, weights)


if __name__ == "__main__":
    '''X = np.array([[-1,-1],[-2,-1], [1,1], [2,1]])
    y = np.array([1,1,2,2])
    clf = SVC()
    clf.fit(X,y)
    print(clf.predict([[-0.8,-1]]))'''
    #clf = SVC()
    clf = SVC(C = 1000, kernel = 'linear')  
    monteavgavgQP = list()
    monteavgavgPLA = list()
    approxavgQP = list()
    vectornumberavg = list()
    predictavg = list()
    for j in range(1):
        #clf = SVC(C = 1000, kernel = 'linear') 
        monteavgQP = list()
        monteavgPLA = list()
        approxQP = list()
        vectoravg = list()
        for k in range(1000):
            nopoints = 100
            line = getRandomLine()
            sample = getSample(line[0], line[1], nopoints)
            #print(sample)
            X = np.c_[sample[:,0], sample[:,1]]
            y = sample[:,2]
            #print(y)
            clf.fit(X,y)
            #print(clf.score(X,y))
            w, it = doPLA(sample)
            #print(len(clf.support_vectors_))
            #print(clf.support_vectors_)
            #print(clf.support_)
            vectoravg.append(len(clf.support_vectors_))
            #print(clf.predict(clf.support_vectors_)==1)
            #print(clf.predict(clf.support_vectors_))
            #print(clf.coef_)
            montelistQP = [doMonteCarloQP(line[0], line[1], clf, 500) for i in range(1)]
            qpMonte = sum(montelistQP)/len(montelistQP)
            monteavgQP.append(sum(montelistQP)/len(montelistQP))
            
            montelist = [ doMonteCarloNP(line[0], line[1], w, 500) for i in range(1)]
            plaMonte = sum(montelist)/len(montelist)
            monteavgPLA.append(plaMonte)
            if(montelistQP < monteavgPLA):
                approxQP.append(1)
            else:
                approxQP.append(0)
            
        #print(sum(monteavgQP)/len(monteavgQP))
        #print(sum(monteavgPLA)/len(monteavgPLA))
        #print(sum(approxQP)/len(approxQP))
        monteavgavgQP.append(sum(monteavgQP)/len(monteavgQP))
        monteavgavgPLA.append(sum(monteavgPLA)/len(monteavgPLA))
        approxavgQP.append(sum(approxQP)/len(approxQP))
        vectornumberavg.append(sum(vectoravg)/len(vectoravg))
    print(sum(monteavgavgQP)/len(monteavgavgQP))
    print(sum(monteavgavgPLA)/len(monteavgavgPLA))
    print("how good is it? ", sum(approxavgQP)/len(approxavgQP))
    print("how good is it? ", sum(vectornumberavg)/len(vectornumberavg))
    
    