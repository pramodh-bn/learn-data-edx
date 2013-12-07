# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:11:46 2013

@author: pramodh
"""

import numpy as np
from sklearn import svm, cluster

def getPoints(numberOfPoints):
    pointList = np.c_[np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints)]
    return pointList

def applyFunction(points):
    return np.sign(points[:,1]-points[:,0]+0.25*np.sin(np.pi * points[:,0]))

def doAssignment13():
    experiments = 1000
    gama = 1.5
    numPoints = 100
    clf = svm.SVC(C= np.inf , kernel="rbf", coef0=1, gamma=gama)
    Ein0 = 0
    for i in range(experiments):    
        X = getPoints(numPoints)
        y = applyFunction(X)
        clf.fit(X,y)
        #print(clf.score(X,y))
        if(1-clf.score(X,y)==0):
            #print("here")
            Ein0 += 1
    print(1-float(Ein0)/experiments)

def doAssignment14():
    gama = 1.5
    numPoints = 100
    k = 9
    experiments = 20
    #km = cluster.KMeans(n_clusters=k, init='k-means++', n_init=5)
    km = cluster.KMeans(n_clusters=k)
    clf = svm.SVC(C= np.inf , kernel="rbf", coef0=1, gamma=gama)
    for j in range(10):
        montelist = []
        for i in range(experiments):
            X = getPoints(numPoints)
            y = applyFunction(X)
            u = km.fit(X).cluster_centers_
            theta = rbfTransform(X, u, gama)
            w = np.linalg.lstsq(theta, y)[0]
            clf.fit(X,y)
            k = [doMonteCarlo(w, clf, 1000, u, gama) for i in range(100)]
            ar = np.array(k)
            #print(float(len(ar)-np.sum(ar[:,0]-ar[:,1] > 0))/len(ar)) 
            montelist.append(float(len(ar)-np.sum(ar[:,0]-ar[:,1] > 0))/len(ar))
        print(np.mean(montelist))
    
def doAssignment15():
    gama = 1.5
    numPoints = 100
    k = 12
    experiments = 20
    #km = cluster.KMeans(n_clusters=k, init='k-means++', n_init=5)
    km = cluster.KMeans(n_clusters=k)
    clf = svm.SVC(C= np.inf , kernel="rbf", coef0=1, gamma=gama)
    for j in range(10):
        montelist = []
        for i in range(experiments):
            X = getPoints(numPoints)
            y = applyFunction(X)
            u = km.fit(X).cluster_centers_
            theta = rbfTransform(X, u, gama)
            w = np.linalg.lstsq(theta, y)[0]
            clf.fit(X,y)
            k = [doMonteCarlo(w, clf, 1000, u, gama) for i in range(100)]
            ar = np.array(k)
            #print(float(len(ar)-np.sum(ar[:,0]-ar[:,1] > 0))/len(ar)) 
            montelist.append(float(len(ar)-np.sum(ar[:,0]-ar[:,1] > 0))/len(ar))
        print(np.mean(montelist))
    
    
def dist2(x):
  return x.T.dot(x)

def rbfTransform(X, U, gamma):
    Fi = np.array([[np.exp(-gamma * dist2(x - u)) for u in U] for x in X])
    #print("Fi", Fi)
    return np.insert(Fi, 0, 1, 1)

def getMisMatches(X, y, weights, centers, gama):
    results = []
    for x in X:
        k = [weights[i] * np.exp(-gama * dist2(x - centers[i-1])) for i in range(1, len(weights))]
        #print(sum(k)+weights[0])
        results.append(np.sign(sum(k)+weights[0]))
    return float(len(X) - np.sum(np.sign(results) == np.sign(y)))/len(X)


def doMonteCarlo(w, clf, numPoints, centers, gama):
    X = getPoints(numPoints)
    y = applyFunction(X)
    eout_hard = 1.0-clf.score(X,y)
    eout_reg = getMisMatches(X, y, w, centers, gama)
    return (eout_hard, eout_reg)

def doMonteCarloReg(w9, w12, numPoints, centers9, centers12,  gama):
    X = getPoints(numPoints)
    y = applyFunction(X)
    return (getMisMatches(X, y, w9, centers9, gama),getMisMatches(X,y,w12,centers12, gama))
    
def doMonteCarloRegGama(w1, w2, numPoints, centers, gama1, gama2):
    X = getPoints(numPoints)
    y = applyFunction(X)
    return (getMisMatches(X, y, w1, centers, gama1),getMisMatches(X,y,w2,centers, gama2))
    

if __name__ == '__main__':
    gama1 = 1.5
    gama2 = 2.0
    numPoints = 100
    experiments = 100
    Ein = []
    Eout = []
    km = cluster.KMeans(n_clusters=9, n_init=1)
    for i in range(experiments):
        X = getPoints(numPoints)
        y = applyFunction(X)
        u = km.fit(X).cluster_centers_
        theta1 = rbfTransform(X, u, gama1)
        theta2 = rbfTransform(X, u, gama2)
        w1 = np.linalg.lstsq(theta1, y)[0]
        w2 = np.linalg.lstsq(theta2, y)[0]
        Ein.append((getMisMatches(X,y,w1, u, gama1), getMisMatches(X,y, w2, u, gama2)))
        kl = [doMonteCarloRegGama(w1, w2, 1000, u, gama1, gama2) for i in range(100)]
        ar = np.array(kl)
        #print(ar)
        Eout.append((np.mean(ar[:,0]), np.mean(ar[:,1])))
        #print(np.mean(montelist))
    '''print(Ein)
    print(Eout)
    for i in range(len(Ein)):
        inE = Ein[i]
        outE = Eout[i]
        strout = ''
        if(inE[0] - inE[1] == 0):
            strout += 'Same,'
        elif(inE[0] - inE[1] > 0):
            strout += 'Down, '
        else:
            strout += 'Up, '
        
        if(outE[0] - outE[1] == 0):
            strout += 'Same'
        elif(outE[0] - outE[1] > 0):
            strout += 'Down'
        else:
            strout += 'Up'
        print(strout)'''
    winnerlist = []
    for i in range(len(Ein)):
        inE = Ein[i]
        outE = Eout[i]
        E_in_12 = inE[1]
        E_in_9 = inE[0]
        E_out_12 = outE[1]
        E_out_9 = outE[0]
        if E_in_12 == E_in_9 and E_out_12 == E_out_9:
            winnerlist.append('e')
        elif E_in_12 < E_in_9 and E_out_12 > E_out_9:
            winnerlist.append('a')
        elif E_in_12 > E_in_9 and E_out_12 < E_out_9:
            winnerlist.append('b')
        elif E_in_12 > E_in_9 and E_out_12 > E_out_9:
            winnerlist.append('c')
        elif E_in_12 < E_in_9 and E_out_12 < E_out_9:
            winnerlist.append('d') 
        else:
            winnerlist.append('f')
            
    print("a", float(winnerlist.count('a'))/len(winnerlist))
    print("b", float(winnerlist.count('b'))/len(winnerlist))
    print("c", float(winnerlist.count('c'))/len(winnerlist))
    print("d", float(winnerlist.count('d'))/len(winnerlist))
    print("e", float(winnerlist.count('e'))/len(winnerlist))
    print("f", float(winnerlist.count('f'))/len(winnerlist))
        
    
        
    

