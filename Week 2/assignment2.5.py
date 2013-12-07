import numpy as np
import random

def sign(value):
    return 1 if value >=0 else -1

def f(x1, x2):
    return sign(((x1*x1) + (x2*x2) - 0.6))

def ga(x1, x2):
    return sign(-1-(0.05*x1) + (0.08*x2) + (0.13*x1*x2) + (1.5*x1*x1) + (1.5*x2*x2))

def gb(x1, x2):
    return sign(-1-(0.05*x1) + (0.08*x2) + (0.13*x1*x2) + (1.5*x1*x1) + (15*x2*x2))

def gc(x1, x2):
    return sign(-1-(0.05*x1) + (0.08*x2) + (0.13*x1*x2) + (15*x1*x1) + (1.5*x2*x2))

def gd(x1, x2):
    return sign(-1-(1.5*x1) + (0.08*x2) + (0.13*x1*x2) + (0.05*x1*x1) + (0.05*x2*x2))

def ge(x1, x2):
    return sign(-1-(0.05*x1) + (0.08*x2) + (1.5*x1*x2) + (0.15*x1*x1) + (0.15*x2*x2))

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
    zlist = [[i[0],i[1], i[2],i[3], i[4], i[5]] for i in dataset]
    ylist = [[i[6]] for i in dataset]
    matX = np.matrix(zlist)
    matY = np.matrix(ylist)
    return np.squeeze(np.asarray(((matX.T * matX).I * matX.T) * matY))
    
def montecarlo(weights, dataset):
    mismatch = 0
    for i in range(len(dataset)):
        y_noise = dataset[i][2]
        valueWithWeight = weights[0] + weights[1] * dataset[i][0] + weights[2] * dataset[i][1] + weights[3] * dataset[i][0] * dataset[i][1] + weights[4] * dataset[i][0] * dataset[i][0] + weights[5] * dataset[i][1] * dataset[i][1]
        #print valueWithWeight, y_noise
        y_weight = sign(valueWithWeight)
        if(y_noise != y_weight):
            mismatch += 1
    #print "mismatches ", mismatch
    return float(mismatch)/len(dataset)

def getmismatches(weights, dataset):
    mismatch = 0
    for i in range(len(dataset)):
        y_noise = dataset[i][6]
        valueWithWeight = weights[0] + weights[1] * dataset[i][1] + weights[2] * dataset[i][2] + weights[3] * dataset[i][3] + weights[4] * dataset[i][4] + weights[5] * dataset[i][5]
        #print valueWithWeight, y_noise
        y_weight = sign(valueWithWeight)
        if(y_noise != y_weight):
            mismatch += 1
    #print "mismatches ", mismatch
    return float(mismatch)/len(dataset)
    
def getmatches(weights, pointset):
    match_a = 0
    match_b = 0
    match_c = 0
    match_d = 0
    match_e = 0
    lengthp = len(pointset)
    for i in pointset:
        valuewithweight = weights[0] + weights[1]*i[0] + weights[2]*i[1] + weights[3]*i[0]*i[1] + weights[4]*i[0]*i[0] + weights[5]*i[1]*i[1]
        y_weight = sign(valuewithweight)
        y_a = ga(i[0],i[1])
        if(y_weight == y_a):
            match_a += 1
        y_b = gb(i[0],i[1])
        if(y_weight == y_b):
            match_b += 1
        y_c = gc(i[0],i[1])
        if(y_weight == y_c):
            match_c += 1
        y_d = gd(i[0],i[1])
        if(y_weight == y_d):
            match_d += 1
        y_e = ge(i[0],i[1])
        if(y_weight == y_e):
            match_e += 1
    return (float(match_a)/lengthp, float(match_b)/lengthp, float(match_c)/lengthp, float(match_d)/lengthp, float(match_e)/lengthp)

def match_calculation():   
    noExperiments = 100
    noPoints = 1000
    matchlist_a = list()
    matchlist_b = list()
    matchlist_c = list()
    matchlist_d = list()
    matchlist_e = list()
    
    for i in range(noExperiments):
        cluster = getPoints(noPoints)
        ylist =  getY(cluster)
        #print "old", ylist
        y_noiselist = addNoise(ylist)
        #print "new", y_noiselist
        zwithy = [[1, cluster[i][0], cluster[i][1], cluster[i][0] * cluster[i][1], cluster[i][0] * cluster[i][0], cluster[i][1] * cluster[i][1], y_noiselist[i]] for i in range(len(cluster))]
        weights = doLinearRegression(zwithy)
        pointset = getPoints(noPoints)
        matches = getmatches(weights, pointset)
        matchlist_a.append(matches[0])
        matchlist_b.append(matches[1])
        matchlist_c.append(matches[2])
        matchlist_d.append(matches[3])
        matchlist_e.append(matches[4])
    #print mismatchlist
    print "average matches a =", sum(matchlist_a)/noExperiments    
    print "average matches b =", sum(matchlist_b)/noExperiments
    print "average matches c =", sum(matchlist_c)/noExperiments
    print "average matches d =", sum(matchlist_d)/noExperiments
    print "average matches e =", sum(matchlist_e)/noExperiments            
    
    
if __name__ == '__main__':
    noExperiments = 1000
    noPoints = 1000
    avgmlist = list()
    for k in range(20):
        mismatchlist = list()
        cluster = getPoints(noPoints)
        ylist =  getY(cluster)
        #print "old", ylist
        y_noiselist = addNoise(ylist)
        #print "new", y_noiselist
        zwithy = [[1, cluster[i][0], cluster[i][1], cluster[i][0] * cluster[i][1], cluster[i][0] * cluster[i][0], cluster[i][1] * cluster[i][1], y_noiselist[i]] for i in range(len(cluster))]
        weights = doLinearRegression(zwithy)
        for i in range(noExperiments):
            newpoints = getPoints(noPoints)
            newy = getY(newpoints)
            newy_noise = addNoise(newy)
            dataset_new = [[newpoints[i][0], newpoints[i][1], newy_noise[i]] for i in range(len(newpoints))]
            mismatch = montecarlo(weights, dataset_new)
            mismatchlist.append(mismatch)
        print "average montecarlo = ", sum(mismatchlist)/noExperiments
        avgmlist.append(sum(mismatchlist)/noExperiments)
    print avgmlist
    print "average of averages ", sum(avgmlist)/len(avgmlist)
    #print mismatchlist
