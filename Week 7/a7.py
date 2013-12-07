import numpy as np

def getWeightsUsingLinearRegressionConstant(in_data):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = np.c_[list1]
    yInputlist = in_data[:,1]
    return np.linalg.lstsq(regInputlist, yInputlist)[0]

def getWeightsUsingLinearRegressionX(in_data):
    list1 = np.empty(len(in_data))
    list1.fill(1)
    regInputlist = np.c_[list1, in_data[:,0]]
    yInputlist = in_data[:,1]
    return np.linalg.lstsq(regInputlist, yInputlist)[0]

def getSquarederrorConst(w, points):
    weights = np.array(w)
    point = np.array(points)
    #print "point is ", point[1]
    return np.square(weights-point[1])

def getSquarederrorX(w, points):
    #list1 = np.empty(len(points))
    #list1.fill(w[0])
    print w[1] * points[0] + w[0]
    results = w[0] + w[1]*points[0]    
    print "this is w ", w, results, points[0], points[1]
    #point = np.array(points)
    #print "point is ", points
    return np.square(results-points[1])

if __name__ == "__main__":
    for et in [np.sqrt(9 + 4 * np.sqrt(6)), 1.0]:#[np.sqrt(np.sqrt(3)+4), np.sqrt(np.sqrt(3)-1), np.sqrt(9 + 4 * np.sqrt(6)),np.sqrt(9 - np.sqrt(6)) ]:
        print et
        points = np.array([[-1.0,0],[et, 1],[1.0, 0]])
        print points
        errorconst = list()
        errorx = list()
        for i in range(len(points)):
            subset = np.delete(points, i, axis=0)
            #print subset, points[i]
            constlist = list()
            xlist = list()
            for k in range(1):
                wconst = getWeightsUsingLinearRegressionConstant(subset)
                wx = getWeightsUsingLinearRegressionX(subset)
                #print wconst, wx
                constlist.append(getSquarederrorConst(wconst, points[i]))
                xlist.append(getSquarederrorX(wx, points[i]))
            print points[i], "error is ", sum(constlist)/len(constlist), sum(xlist)/len(xlist)
            errorconst.append(sum(constlist)/len(constlist))
            errorx.append(sum(xlist)/len(xlist))
        print sum(errorconst)/3.0, sum(errorx)/3.0 
    