import numpy as np
import matplotlib.pyplot as plt

def getWeightsUsingLinearRegression(in_data):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2, in_data[:,1]**2, in_data[:,0] * in_data[:,1], np.absolute(in_data[:,0] - in_data[:,1]), np.absolute(in_data[:,0] + in_data[:,1])]
    yInputlist = in_data[:,2]
    return np.linalg.lstsq(regInputlist, yInputlist)[0]

def getRegularisedLinearRegression(in_data, lbda):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2, in_data[:,1]**2, in_data[:,0] * in_data[:,1], np.absolute(in_data[:,0] - in_data[:,1]), np.absolute(in_data[:,0] + in_data[:,1])]
    yInputlist = np.c_[in_data[:,2]]

    matX = np.matrix(regInputlist)
    matY = np.matrix(yInputlist)
    #print matX
    #print matY
    #print lbda * np.identity(matX.shape[1])
    return np.squeeze(np.asarray(((matX.T * matX + lbda * np.identity(matX.shape[1])).I * matX.T) * matY))
    
    #return np.linalg.lstsq(regInputlist, yInputlist)[0]
def sign(x):
    return 1 if x >= 0 else -1
    
def getMisMatches2(data, weights):
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2+weights[5]*data[:,0] * data[:,1]+weights[6]*np.absolute(data[:,0] - data[:,1])+weights[7]*np.absolute(data[:,0] + data[:,1])
    mismatches = 0
    for i in range(len(results)):
        if sign(results[i]) != data[:,2][i]:
            mismatches = mismatches + 1
    return float(mismatches)/len(data)

def getMisMatches1(data, weights):
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2+weights[5]*data[:,0] * data[:,1]+weights[6]*np.absolute(data[:,0] - data[:,1])+weights[7]*np.absolute(data[:,0] + data[:,1])
    #print np.sign(results)
    #print np.sign(data[:,2])
    #print "are we here ", countneg, origneg
    #print np.sign(results) == np.sign(data[:,2])
    #print np.sum(np.sign(results) == np.sign(data[:,2]))
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)

def getMisMatches(data, weights):
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2+weights[5]*data[:,0] * data[:,1]+weights[6]*np.absolute(data[:,0] - data[:,1])+weights[7]*np.absolute(data[:,0] + data[:,1])
    countneg = len(np.extract(np.sign(results) == -1, results))
    origneg  = len(np.extract(np.sign(data[:,2])== -1,data[:,2]))
    #print countneg, origneg
    return float(np.absolute(countneg-origneg))/len(data)

def assignment2(in_data, out_data):
    w = getWeightsUsingLinearRegression(in_data)
    #print "mismatches input ", getMisMatches1(in_data, w), getMisMatches(in_data, w)
    #print "mismatches output ", getMisMatches1(out_data, w), getMisMatches(out_data, w)
    errorvec = np.array([getMisMatches1(in_data, w),getMisMatches1(out_data, w)])
    options = np.array([[0.03, 0.08], [0.03, 0.10], [0.04, 0.09], [0.04, 0.11], [0.05, 0.10]])
    return np.sqrt(((errorvec-options[:])**2)[:,0]+((errorvec-options[:])**2)[:,1])
    
def assignment3(in_data, out_data, k, options):
    w = getRegularisedLinearRegression(in_data, 10**k)
    #print "mismatches input ", getMisMatches1(in_data, w), getMisMatches(in_data, w)
    #print "mismatches output ", getMisMatches1(out_data, w), getMisMatches(out_data, w)
    errorvec = np.array([getMisMatches1(in_data, w),getMisMatches1(out_data, w)])
    return np.sqrt(((errorvec-options[:])**2)[:,0]+((errorvec-options[:])**2)[:,1])

def assignment5(in_data, out_data, koptions):
    #return [getMisMatches1(out_data, getRegularisedLinearRegression(in_data, 10**i)) for i in koptions]
    retlist = list()
    for i in koptions:
        w = getRegularisedLinearRegression(in_data, 10**i)
        print "mismatches output for ", i, w#getMisMatches2(out_data, w), getMisMatches1(out_data, w), getMisMatches(out_data, w)        
        retlist.append(getMisMatches1(out_data, w))
    return retlist
    #return [getMisMatches(out_data, w + (10**i/len(out_data))*wsoft) for i in koptions]
    
    


if __name__ == "__main__":
    in_data = np.genfromtxt("C:\Study\EdX\Learning From Data\Week 6\Assignment\in.data", dtype=float)
    out_data = np.genfromtxt("C:\Study\EdX\Learning From Data\Week 6\Assignment\out.data", dtype=float)
    #w = getWeightsUsingLinearRegression(in_data)
    #print(assignment2(in_data, out_data))
    #print(assignment3(in_data, out_data, -3, np.array([[0.01, 0.02], [0.02, 0.04], [0.02, 0.06], [0.03, 0.08], [0.03, 0.10]])))
    #print(assignment3(in_data, out_data, 3, np.array([[0.2, 0.2], [0.2, 0.3], [0.3, 0.3], [0.3, 0.4], [0.4, 0.4]])))
    #print assignment5(in_data, out_data, [2, 1, 0, -1, -2])
    #plt.plot( [2, 1, 0, -1, -2], assignment5(in_data, out_data, [2, 1, 0, -1, -2]), c="r")
    #print zip(np.arange(-2,2.5, 0.1), assignment5(in_data, out_data, np.arange(-2,2.5, 0.1)))
    #plt.plot(np.arange(-50,50,1.0), assignment5(in_data, out_data, np.arange(-50,50,1.0)))
    #plt.show()
    #print assignment5(in_data, out_data, [1, 0, -1])
    l = np.arange(-10, 10, 1.0)
    print assignment5(in_data, out_data, l)
    

