# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:35:06 2013

@author: dyanna
"""

import numpy as np
import scipy as sp
from sklearn.svm import SVC
from operator import itemgetter


def classifyPoints(num, sample, vs=-1.0):
    if vs == -1.0:
        # vs all
        negatives = sample[sample[:,0] != num]
    else:
        negatives = sample[sample[:,0] == vs]

    positives = sample[sample[:,0] == num]
    listpositive = np.empty(len(positives))
    listpositive.fill(1.0)
    datapositive = np.c_[positives[:,1],positives[:,2], listpositive]
    negativelist = np.empty(len(negatives))
    negativelist.fill(-1.0)
    datanegative = np.c_[negatives[:,1], negatives[:,2], negativelist]
    newsample = np.concatenate((datanegative, datapositive), axis=0)
    np.random.shuffle(newsample)
    return newsample

def predictEinRBF(in_data, num, c, vs=-1.0): 
    sample = classifyPoints(num, in_data, vs)
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    clf = SVC(C = c, kernel = 'rbf', gamma=1.0)  
    clf.fit(X,y)
    #print(1-clf.score(X,y))
    #print(float(len(sample) - np.sum(np.sign(clf.predict(X)) == np.sign(sample[:,2])))/len(sample))
    return 1-clf.score(X,y)

def predictEoutRBF(in_data, out_data, num, c, vs=-1.0): 
    sample = classifyPoints(num, in_data, vs)
    out_sample = classifyPoints(num, out_data, vs)
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    clf = SVC(C = c, kernel = 'rbf', gamma=1.0)  
    clf.fit(X,y)
    Xtest = np.c_[out_sample[:,0], out_sample[:,1]]
    ytest = out_sample[:,2]
    #print("what SVC gets ", 1-clf.score(Xtest,ytest))
    #print("What I get ", float(len(out_sample) - np.sum(np.sign(clf.predict(Xtest)) == np.sign(out_sample[:,2])))/len(out_sample))
    return 1-clf.score(Xtest,ytest)


def predictEin(in_data, num, c, deg, vs=-1.0): 
    sample = classifyPoints(num, in_data, vs)
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1, gamma=1.0)  
    #clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0)
    clf.fit(X,y)
    #print(1-clf.score(X,y))
    #print(float(len(sample) - np.sum(np.sign(clf.predict(X)) == np.sign(sample[:,2])))/len(sample))
    return (num, 1-clf.score(X,y)) 

def predictEout(in_data, out_data, num, c, deg, vs=-1.0): 
    sample = classifyPoints(num, in_data, vs)
    out_sample = classifyPoints(num, out_data, vs)
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0, gamma=1.0)  
    #clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0)
    clf.fit(X,y)
    Xtest = np.c_[out_sample[:,0], out_sample[:,1]]
    ytest = out_sample[:,2]
    #print("what SVC gets ", 1-clf.score(Xtest,ytest))
    #print("What I get ", float(len(out_sample) - np.sum(np.sign(clf.predict(Xtest)) == np.sign(out_sample[:,2])))/len(out_sample))
    return 1-clf.score(Xtest,ytest)

def predictEValid(sample, valid_sample, num, c, deg, vs=-1.0): 
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    #print(y)
    #clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0, gamma=1.0)  
    clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0)
    clf.fit(X,y)
    Xtest = np.c_[valid_sample[:,0], valid_sample[:,1]]
    ytest = valid_sample[:,2]
    #print("what SVC gets ", 1-clf.score(Xtest,ytest))
    #print("What I get ", float(len(out_sample) - np.sum(np.sign(clf.predict(Xtest)) == np.sign(out_sample[:,2])))/len(out_sample))
    return 1-clf.score(Xtest,ytest)


def getSupportVectors(in_data, num, c, deg, vs=-1.0):
    #print(num)
    sample = classifyPoints(num, in_data, vs)
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0, gamma=1.0)  
    #clf = SVC(C = c, kernel = 'poly', degree=deg, coef0=1.0)
    clf.fit(X,y)
    #print(clf.support_vectors_)   
    #print(len(clf.support_vectors_))
    #print(clf.n_support_)
    return len(clf.support_vectors_)

def doAssignment234(in_data):
    c = 0.01
    q = 2
    numlist_1 = [0.0, 2.0, 4.0, 6.0, 8.0]
    q3 = []
    
    k = [predictEin(in_data, i, c, q) for i in numlist_1]
    print(k)
    print("answer for 2 ", max(k, key=itemgetter(1)))
    q3.append(max(k, key=itemgetter(1)))
    
    numlist_2 = [1.0, 3.0, 5.0, 7.0, 9.0]
    l = [predictEin(in_data, i, c, q) for i in numlist_2]
    print(l)
    print("answer for 3 ", min(l, key=itemgetter(1)))
    q3.append(min(l, key=itemgetter(1)))
    
    #q3 = [(1.0,9.0)]
    m = [getSupportVectors(in_data, i[0], c, q) for i in q3]
    print(m)
    diff = abs(m[0]-m[1])
    print(diff)
    answers = [600, 1200, 1800, 2400, 3000]
    n = [(i, abs(diff-i)) for i in answers]
    print("answer for 4 ", min(n,key=itemgetter(1)))

def doAssignment5(in_data, out_data):
    c = [0.001,0.01, 0.1, 1.0]
    q = 2
    num = 1.0
    vs = 5.0
    svs = [(i, getSupportVectors(in_data, num, i, q, vs)) for i in c]
    print("option a ", svs)
    eouts = [(i, predictEout(in_data, out_data, num, i, q, vs)) for i in c]
    print("option c ", eouts)
    eins = [(i, predictEin(in_data, num, i, q, vs)[1]) for i in c]
    print("option d ", eins)
    
def doAssignment6(in_data, out_data):
    num = 1.0
    vs = 5.0
    c = 0.0001
    q = [2,5]
    eins = [(i, predictEin(in_data,num,c,i,vs)[1]) for i in q]
    print("option a ",eins)
    c = 0.001
    svs = [(i, getSupportVectors(in_data, num, c, i, vs)) for i in q]
    print("option b ",svs)
    c = 0.01
    eins = [(i, predictEin(in_data,num,c,i,vs)[1]) for i in q]
    print("option c ",eins)
    c = 1.0
    eouts = [(i, predictEout(in_data, out_data, num, c, i, vs)) for i in q]
    print("option d ", eouts)    

def getPartitions(parts):
    partitions = []
    for i in range(len(parts)):
        print(i)
        valid_data = parts[i]
        train_data = np.concatenate([parts[j] for j in range(len(parts)) if j != i])
        print(len(valid_data))
        print(len(train_data))
        partitions.append((valid_data, train_data))
    return partitions

def isPartitionValid(partitions):
    for partition in partitions:
        y = partition[0][:,2]
        if(len(y[y==-1]) == 0 or len(y[y==1]) == 0):
            return False
        y1 = partition[1][:,2]
        if(len(y1[y1==-1]) == 0 or len(y1[y1==1]) == 0):
            return False
        
    return True


def partitionSample(sample, size):
    np.random.shuffle(sample)
    parts = np.array_split(sample, size)
    partitions = []
    for i in range(len(parts)):
        #print(i)
        valid_data = parts[i]
        train_data = np.concatenate([parts[j] for j in range(len(parts)) if j != i])
        #print(len(valid_data))
        #print(len(train_data))
        partitions.append((valid_data, train_data))
    return np.array(partitions)
    
def doAssignment78(in_data):
    size = 10
    num = 1.0
    vs = 5.0
    deg = 2
    clist = [0.0001, 0.001, 0.01, 0.1, 1.0]
    sample = classifyPoints(num, in_data, vs)
    avgofavg = []
    for r in range(3):
        winnerlist = []
        winnerEcv = []
        for i in range(100):
            rlist = []
            for c in clist:
                m = partitionSample(sample, size)
                eouts = [predictEValid(j[1], j[0], num, c, deg, vs) for j in m]
                eoutswinner = [predictEValid(j[1], j[0], num, 0.01, deg, vs) for j in m]
                rlist.append((c, np.mean(eouts)))
                winnerEcv.append(np.mean(eoutswinner))
            #print(rlist)
            winnerlist.append(min(rlist, key=itemgetter(1))[0])
            #winnerEcv.append(min(rlist, key=itemgetter(1))[1])
        #print(winnerlist)
        print(sp.stats.itemfreq(winnerlist))
        print(np.mean(winnerEcv))
        avgofavg.append(np.mean(winnerEcv))
    print( avgofavg)
    

def doAssignment910(in_data, out_data):
    clist = [0.001, 0.01, 1, 10, 100, 10**4, 10**6]
    num = 1.0
    vs = 5.0
    eins = [(i, predictEinRBF(in_data, num, i, vs)) for i in clist]
    print(eins)
    print(min(eins, key=itemgetter(1)))
    eouts = [(i, predictEoutRBF(in_data, out_data, num, i, vs)) for i in clist]    
    print(eouts)
    print(min(eouts, key=itemgetter(1)))
    

if __name__ == "__main__":
    in_data = np.genfromtxt("features.train", dtype=float)
    out_data = np.genfromtxt("features.test", dtype=float)
    #doAssignment234(in_data)  
    #doAssignment5(in_data, out_data)
    #doAssignment6(in_data, out_data)
    #doAssignment78(in_data)
    #doAssignment910(in_data, out_data)
    

    
    
        
    

    
    
    