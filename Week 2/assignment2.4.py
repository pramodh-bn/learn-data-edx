import numpy as np
import random

def sign(value):
    return 1 if value >=0 else -1

def f(x1, x2):
    return sign(((x1*x1) + (x2*x2) - 0.6))

def getPoints(numberOfPoints):
    pointList = zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints))
    return pointList

def getY(points):
    return [f(i[0], i[1]) for i in points]

def addNoise(ylist):
    randomlist = random.sample(range(len(ylist)), int(0.1*noPoints))
    #print randomlist
    for i in randomlist:
        ylist[i] = -1 * ylist[i]
    return ylist
    
def doLinearRegression(dataset):
    xlist = [[i[0],i[1], i[2]] for i in dataset]
    ylist = [[i[3]] for i in dataset]
    matX = np.matrix(xlist)
    matY = np.matrix(ylist)
    return np.squeeze(np.asarray(((matX.T * matX).I * matX.T) * matY))
    
def getmismatches(weights, dataset):
    mismatch = 0
    for i in range(len(dataset)):
        y_noise = dataset[i][3]
        valueWithWeight = weights[0] + weights[1] * dataset[i][1] + weights[2] * dataset[i][2]
        #print valueWithWeight, y_noise
        y_weight = sign(valueWithWeight)
        if(y_noise != y_weight):
            mismatch += 1
    #print "mismatches ", mismatch
    return float(mismatch)/len(dataset)
    

if __name__ == '__main__':
    noExperiments = 1000
    noPoints = 1000
    mismatchlist = list()
    for i in range(noExperiments):
        cluster = getPoints(noPoints)
        ylist =  getY(cluster)
        #print "old", ylist
        y_noiselist = addNoise(ylist)
        #print "new", y_noiselist
        xwithy = [[1, cluster[i][0], cluster[i][1], y_noiselist[i]] for i in range(len(cluster))]
        weights = doLinearRegression(xwithy)
        mismatches = getmismatches(weights, xwithy)
        mismatchlist.append(mismatches)
    #print mismatchlist
    print "average misses=", sum(mismatchlist)/noExperiments
            
        
    
    
    