# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:15:29 2013

@author: pramodh
"""
import numpy as np
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


def getRegularisedLinearRegression(in_data, lbda):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = np.c_[list1, in_data[:,0], in_data[:,1]]
    yInputlist = np.c_[in_data[:,2]]

    matX = np.matrix(regInputlist)
    matY = np.matrix(yInputlist)
    #print matX
    #print matY
    #print lbda * np.identity(matX.shape[1])
    return np.squeeze(np.asarray(((matX.T * matX + lbda * np.identity(matX.shape[1])).I * matX.T) * matY))

def getRegularisedLinearRegressionTransform(in_data, lbda):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = np.c_[list1, in_data[:,0], in_data[:,1],in_data[:,0] * in_data[:,1], in_data[:,0]**2, in_data[:,1]**2]
    yInputlist = np.c_[in_data[:,2]]

    matX = np.matrix(regInputlist)
    matY = np.matrix(yInputlist)
    #print matX
    #print matY
    #print lbda * np.identity(matX.shape[1])
    return np.squeeze(np.asarray(((matX.T * matX + lbda * np.identity(matX.shape[1])).I * matX.T) * matY))

def getMisMatchesTransform(data, weights):
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0] * data[:,1]+weights[4]*data[:,0]**2+weights[5]*data[:,1]**2
    #print np.sign(results)
    #print np.sign(data[:,2])
    #print "are we here ", countneg, origneg
    #print np.sign(results) == np.sign(data[:,2])
    #print np.sum(np.sign(results) == np.sign(data[:,2]))
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)

def getMisMatches(data, weights):
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]
    #print np.sign(results)
    #print np.sign(data[:,2])
    #print "are we here ", countneg, origneg
    #print np.sign(results) == np.sign(data[:,2])
    #print np.sum(np.sign(results) == np.sign(data[:,2]))
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)


def doAssignment7(in_data):
    numlist = [5.0, 6.0, 7.0, 8.0, 9.0]
    lbda = 1.0
    k = []
    for num in numlist:
        sample = classifyPoints(num, in_data)
        k.append((num, getMisMatches(sample, getRegularisedLinearRegression(sample, lbda))))
    print(k)
    print(min(k, key=itemgetter(1)))        

def doAssignment8(in_data, out_data):
    numlist = [0.0, 1.0, 2.0, 3.0, 4.0]
    lbda = 1.0
    k = []
    for num in numlist:
        in_sample = classifyPoints(num, in_data)
        test_sample = classifyPoints(num, out_data)
        k.append((num, getMisMatchesTransform(test_sample, getRegularisedLinearRegressionTransform(in_sample, lbda))))
    print(k)
    print(min(k, key=itemgetter(1)))        
    
def doAssignment9(in_data, out_data):
    numlist = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    lbda = 1.0
    EinsNormal = []
    EinsTransform = []
    EoutsNormal = []
    EoutsTransform = []
    for num in numlist:
        in_sample = classifyPoints(num, in_data)
        test_sample = classifyPoints(num, out_data)
        EinsNormal.append(getMisMatches(in_sample, getRegularisedLinearRegression(in_sample, lbda)))
        EinsTransform.append(getMisMatchesTransform(in_sample, getRegularisedLinearRegressionTransform(in_sample, lbda)))
        EoutsNormal.append(getMisMatches(test_sample, getRegularisedLinearRegression(in_sample, lbda)))
        EoutsTransform.append(getMisMatchesTransform(test_sample, getRegularisedLinearRegressionTransform(in_sample, lbda)))
    #print(num)
    #print(EinsNormal)
    #print(EinsTransform)
    #print(EoutsNormal)
    #print(EoutsTransform)
    fulldata = zip(numlist, EinsNormal, EinsTransform, EoutsNormal, EoutsTransform)
    for i in fulldata:
        if(i[0] == 5):
            print(i[3], i[4], i[4]-i[3])

def doAssignment10(in_data, out_data):
    num = 1.0
    vs = 5.0 
    in_sample = classifyPoints(num, in_data, vs)
    test_sample = classifyPoints(num, out_data, vs)
    #print(len(in_sample))
    lbdalist = [1, 0.01]
    EinsTransform = []
    EoutsTransform = []
    
    for lbda in lbdalist:
        EinsTransform.append(getMisMatchesTransform(in_sample, getRegularisedLinearRegressionTransform(in_sample, lbda)))
        EoutsTransform.append(getMisMatchesTransform(test_sample, getRegularisedLinearRegressionTransform(in_sample, lbda)))
    
    fulldata = zip(lbdalist, EinsTransform, EoutsTransform)
    for i in fulldata:
        #print(i[0],i[1],i[2],i[3],i[4])
        print(i[0], i[1], i[2])
    
    

if __name__ == "__main__":
    in_data = np.genfromtxt("features.train", dtype=float)
    out_data = np.genfromtxt("features.test", dtype=float)
    doAssignment10(in_data, out_data)
    
        

