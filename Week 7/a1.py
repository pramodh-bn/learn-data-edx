import numpy as np

def getWeightsUsingLinearRegression(in_data, k):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = list()
    if k == 1:
        regInputlist = np.c_[list1, in_data[:,0]]
    elif k == 2:
        regInputlist = np.c_[list1, in_data[:,0], in_data[:,1]]
    elif k == 3:
        regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2]
    elif k == 4:
        regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2, in_data[:,1]**2]
    elif k == 5:
        regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2, in_data[:,1]**2, in_data[:,0] * in_data[:,1]]
    elif k == 6:
        regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2, in_data[:,1]**2, in_data[:,0] * in_data[:,1], np.absolute(in_data[:,0] - in_data[:,1])]
    else:
        regInputlist = np.c_[list1, in_data[:,0], in_data[:,1], in_data[:,0]**2, in_data[:,1]**2, in_data[:,0] * in_data[:,1], np.absolute(in_data[:,0] - in_data[:,1]), np.absolute(in_data[:,0] + in_data[:,1])]

    yInputlist = in_data[:,2]
    return np.linalg.lstsq(regInputlist, yInputlist)[0]

def getMisMatches(data, weights, k):
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = 0
    if k == 1:
        results = list1 + weights[1]*data[:,0]
    elif k == 2:
        results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]
    elif k == 3:
        results = list1 + weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2
    elif k == 4:
        results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2
    elif k == 5:
        results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2+weights[5]*data[:,0] * data[:,1]
    elif k == 6:
        results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2+weights[5]*data[:,0] * data[:,1]+weights[6]*np.absolute(data[:,0] - data[:,1])
    else:
        results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]+weights[3]*data[:,0]**2+weights[4]*data[:,1]**2+weights[5]*data[:,0] * data[:,1]+weights[6]*np.absolute(data[:,0] - data[:,1])+weights[7]*np.absolute(data[:,0] + data[:,1])
    #print np.sign(results)
    #print np.sign(data[:,2])
    #print "are we here ", countneg, origneg
    #print np.sign(results) == np.sign(data[:,2])
    #print np.sum(np.sign(results) == np.sign(data[:,2]))
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)

def assignment1(train_data, valid_data):
    mismatchlist = list()
    for k in range(3,8):
        print(k)
        w = getWeightsUsingLinearRegression(train_data, k)
        mismatchlist.append(getMisMatches(valid_data, w, k))
    print(mismatchlist)

def assignment2(train_data, out_data):    
    mismatchlist = list()
    for k in range(3,8):
        print(k)
        w = getWeightsUsingLinearRegression(train_data, k)
        mismatchlist.append(getMisMatches(out_data, w, k))
    print(mismatchlist)

def assignment3(train_data, valid_data):
    mismatchlist = list()
    for k in range(3,8):
        print(k)
        w = getWeightsUsingLinearRegression(train_data, k)
        mismatchlist.append(getMisMatches(valid_data, w, k))
    print(mismatchlist)

def assignment4(train_data, out_data):    
    mismatchlist = list()
    for k in range(3,8):
        print(k)
        w = getWeightsUsingLinearRegression(train_data, k)
        mismatchlist.append(getMisMatches(out_data, w, k))
    print(mismatchlist)

if __name__ == "__main__":
    in_data = np.genfromtxt("C:\Study\EdX\Learning From Data\Week 6\Assignment\in.data", dtype=float)
    out_data = np.genfromtxt("C:\Study\EdX\Learning From Data\Week 6\Assignment\out.data", dtype=float)
    print(len(in_data), len(in_data[:25]), len(in_data[25:]))
    train_data = in_data[:25]
    valid_data = in_data[25:]
    #assignment2(train_data, out_data)
    #valid_data = in_data[:25]
    #train_data = in_data[25:]
    #assignment4(train_data, out_data)
    mismatchlistein = list()
    mismatchlisteval = list()
    mismatchlisteout = list()
    for k in range(1,8):
        #print k
        w = getWeightsUsingLinearRegression(train_data, k)
        print("weights ", w, len(w), k)
        mismatchlistein.append(getMisMatches(train_data, w, k))
        mismatchlisteval.append(getMisMatches(valid_data, w, k))
        mismatchlisteout.append(getMisMatches(out_data, w, k))
    print(mismatchlistein)
    print(mismatchlisteval)
    print(mismatchlisteout)
        
        
