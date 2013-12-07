import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines as mpl_lines


def slope_from_points(point1, point2):
    return (point2[1] - point1[1])/(point2[0] - point1[0])

def plot_secant(point1, point2, ax):
    # plot the secant
    slope = slope_from_points(point1, point2)
    intercept = point1[1] - slope*point1[0] 
    # update the points to be on the axes limits
    x = ax.get_xlim()
    y = ax.get_ylim()
    data_y = [x[0]*slope+intercept, x[1]*slope+intercept]
    line = mpl_lines.Line2D(x, data_y, color='red')
    ax.add_line(line)
    #return ax.figure()

def isLeft(a, b, c):
    if ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0:
        return 1
    else:
	return -1

def isLeft1(a, b, c):
    if ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0:
        return 1
    else:
	return 0

def getYval(y):
    return 1 if y>0 else -1

def getRandomLine():
    return zip(np.random.uniform(-1,1.00,2),np.random.uniform(-1,1.00,2))

def getPoints(numberOfPoints):
    pointList = zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints))
    return pointList
    
def plotMonteData(dataset_D, ax):
    xpos = [(i[0], i[1]) for i in dataset_D if i[2] == 1]
    xneg = [(i[0], i[1]) for i in dataset_D if i[2] == -1]
    ax.plot(zip(*xpos)[0], zip(*xpos)[1], "x")
    ax.plot(zip(*xneg)[0], zip(*xneg)[1], "o")

def plotDataSet(dataset_D, pointa, pointb, ax):
    xpos = [(i[0], i[1]) for i in dataset_D if i[2] == 1]
    xneg = [(i[0], i[1]) for i in dataset_D if i[2] == -1]
    ax.plot(zip(*xpos)[0], zip(*xpos)[1], "x")
    ax.plot(zip(*xneg)[0], zip(*xneg)[1], "o")
    plot_secant(pointa, pointb, ax)

def gradient(dataset_D):
    data = dataset_D[np.random.randint(0,100)]
    print data
    w = np.array([1,1,1])
    xdata = np.array([1,data[0], data[1]])
    print xdata
    gradient = np.multiply(data[2], xdata)
    grad = np.dot(np.multiply(data[2],w),xdata)
    print grad
    print "see ", data[2] * np.dot(w, xdata)
    print 1 + np.exp(grad)
    print 0.01 * gradient/grad
        
def euclidNorm(a):
    #print "np dot ",np.dot(a,a)
    #print "np sqrt ", np.sqrt(np.dot(a,a))
    return np.sqrt(np.dot(a,a))

#def doMonteCarlo(pointa, pointb, weight, nopoint, ax):
def doMonteCarlo(pointa, pointb, weight, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    dataset_D = [(i[0],i[1], isLeft(pointa,pointb,i)) for i in points]
    mismatches = 0
    datasetList = []
    for i in dataset_D:
	yy = weight[0] + weight[1] * i[0] + weight[2] * i[1]
	datasetList.append((i[0],i[1], getYval(yy)))
	if(getYval(yy) != i[2]):
            mismatches = mismatches + 1
    #print("mismatches ", mismatches)
    #plotMonteData(datasetList, ax)
    return float(mismatches)/nopoint

#def doMonteCarloReg(pointa, pointb, weight, nopoint, ax):
def doMonteCarloReg(pointa, pointb, weight, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    #dataset_D = [(i[0],i[1], weight[0]+weight[1]*i[0]+weight[2]*i[1]) for i in points]
    dataset_D = [(i[0],i[1], isLeft(pointa, pointb, i)) for i in points]
    outerrx = [np.log((1.+np.exp(-1.*np.dot(weight,np.array([1, data[0], data[1]])*data[2])))) for data in dataset_D]
    #print("mismatches ", mismatches)
    #plotMonteData(datasetList, ax)

    return sum(outerrx)/float(nopoint)   
    

def doExperiment(numberPoints):
    cluster = getPoints(nopoints)
    line = getRandomLine()
    dataset_D = [(i[0],i[1], isLeft(line[0],line[1],i)) for i in cluster]
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plotDataSet(dataset_D, line[0], line[1], ax)
    #original = dataset_D
    #print dataset_D
    
    # gradient descent for some random point
    eta = 0.01
    w = np.array([0,0,0])
    itera = 0
    while True:
        itera += 1
        np.random.shuffle(dataset_D)
        watbegin = w
        for data in dataset_D:
            xdata = np.array([1,data[0], data[1]])
            numerator = -np.multiply(data[2], xdata)
            denom = 1 + np.exp(data[2] * np.dot(w, xdata))
            w = w - eta * numerator/denom
        #print w
        #print "diff" , w-watbegin
        #print "euclid norm ", euclidNorm(w-watbegin)
        if(euclidNorm(w-watbegin) < 0.01):
            break
    #print w, itera
    #montelist = [doMonteCarlo(line[0], line[1], w, 100, ax) for i in range(10)] 
    #montelist_log = [doMonteCarloReg(line[0], line[1], w, 100, ax) for i in range(10)] 
    montelist = [doMonteCarlo(line[0], line[1], w, 100) for i in range(100)] 
    montelist_log = [doMonteCarloReg(line[0], line[1], w, 1000) for i in range(1)] 
    monte = sum(montelist)/len(montelist)
    montereg = sum(montelist_log)/len(montelist_log)
    #print "monte ", monte
    return (monte,montereg, itera)
    
if __name__ == "__main__":
    nopoints = 100
    '''cluster = getPoints(nopoints)
    line = getRandomLine()
    dataset_D = [(i[0],i[1], isLeft(line[0],line[1],i)) for i in cluster]
    original = dataset_D
    #print dataset_D
    
    # gradient descent for some random point
    eta = 0.01
    w = np.array([0,0,0])
    itera = 0
    while True:
        itera += 1
        np.random.shuffle(dataset_D)
        watbegin = w
        for data in dataset_D:
            xdata = np.array([1,data[0], data[1]])
            numerator = -np.multiply(data[2], xdata)
            denom = 1 + np.exp(data[2] * np.dot(w, xdata))
            w = w - eta * numerator/denom
        #print w
        #print "diff" , w-watbegin
        #print "euclid norm ", euclidNorm(w-watbegin)
        if(euclidNorm(w-watbegin) < 0.01):
            break
    print w, itera
    montelist = [doMonteCarlo(line[0], line[1], w, 10000) for i in range(10)] 
    print "monte ", sum(montelist)/len(montelist)'''
    for k in range(2):
        l = [doExperiment(nopoints) for m in range(100)]
        montelist, montelistreg, itera = zip(*l)
        print("monte style ", sum(montelist)/len(montelist))
        print("monte style reg", sum(montelistreg)/len(montelistreg))
        print("avg iteration ", sum(itera)/len(itera))
    #plt.show()
        
    
            
    
