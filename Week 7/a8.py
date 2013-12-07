import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines as mpl_lines

def getRandomLine():
    return zip(np.random.uniform(-1,1.00,2),np.random.uniform(-1,1.00,2))

def getPoints(numberOfPoints):
    pointList = zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints))
    return pointList

def isLeft(a, b, c):
	return 1 if ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0 else -1;

def sign(x):
    return 1 if x > 0 else -1 

def sign1(x):
    return -1 if x > 0 else 1 

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

def getMisMatches1(data, weights):
    #print data
    list1 = np.empty(len(data))
    list1.fill(weights[0])
    results = list1+ weights[1]*data[:,0]+weights[2]*data[:,1]
    print results
    return float(len(data) - np.sum(np.sign(results) == np.sign(data[:,2])))/len(data)

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

def doMonteCarlo(pointa, pointb, weight, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    dataset_D = [(i[0],i[1], isLeft(pointa,pointb,i)) for i in points]
    dataset = np.array(dataset_D)
    mismatches = 0
    datasetList = []
    for i in dataset_D:
	yy = weight[0] + weight[1] * i[0] + weight[2] * i[1]
	datasetList.append((i[0],i[1], sign1(yy)))
	if(sign1(yy) != i[2]):
            mismatches = mismatches + 1
    #print("mismatches ", mismatches)
    #plotMonteData(datasetList, ax)
    #xpos = [(i[0], i[1]) for i in dataset_D if i[2] == 1]
    #xneg = [(i[0], i[1]) for i in dataset_D if i[2] == -1]
    xpos = dataset[dataset[:,2] == 1]
    xpos1 = np.array([(i[0], i[1], 1) for i in dataset if i[2] == 1]) 
    print len(xpos), len(xpos1)
    xneg = dataset[dataset[:,2] == -1]
    #plt.gca().plot(zip(*xpos)[0], zip(*xpos)[1], "+")
    #plt.gca().plot(zip(*xneg)[0], zip(*xneg)[1], "*")
    #plt.plot(xpos[:,0], xpos[:,1], "+")
    #plt.plot(xneg[:,0], xneg[:,1], "*")
    #plt.show()
    return float(mismatches)/nopoint

                
def plotData(sample, line, w):
    xpos = sample[sample[:,2] == 1]
    xneg = sample[sample[:,2] == -1]
    plt.plot(xpos[:,0], xpos[:,1], "x")
    plt.plot(xneg[:,0], xneg[:,1], "o")
    slope = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
    intercept = line[0][1] - slope*line[0][0] 
    # update the points to be on the axes limits
    x = plt.gca().get_xlim()
    #y = plt.gca().get_ylim()
    data_y = [x[0]*slope+intercept, x[1]*slope+intercept]
    line = mpl_lines.Line2D(x, data_y, color='red')
    
    plaPoint1 = (0.1,-((w[1]/w[2])*0.1) + (-(w[0]/w[2])))
    plaPoint2 = (0.8, -((w[1]/w[2])*0.8) + (-(w[0]/w[2])))
    slopePLA = (plaPoint2[1] - plaPoint1[1])/(plaPoint2[0] - plaPoint1[0])
    interceptPLA = plaPoint1[1] - slope*plaPoint1[0] 
    xPLA = plt.gca().get_xlim()
    data_yPLA = [xPLA[0]*slopePLA+interceptPLA, xPLA[1]*slopePLA+interceptPLA]
    linePLA = mpl_lines.Line2D(xPLA, data_yPLA, color='blue')
    
    plt.gca().add_line(line)
    plt.gca().add_line(linePLA)
    #plt.show()


if __name__ == "__main__":
    avgofavgiters = list()
    avgofavgprob = list()
    nopoints = 300
    montel = list()
    iteravg = list()
    for k in range(100):
        cluster = getPoints(nopoints)
        line = getRandomLine()
        sample = np.array([(i[0], i[1], isLeft(line[0], line[1], i)) for i in cluster])
        #plotData(sample, line)
        
        w, it = doPLA(sample)
        #print(getMisMatches(sample, w))
        #plotData(sample, line, w)
        montelist = [ doMonteCarloNP(line[0], line[1], w, 100) for i in range(100)]
        #print sum(montelist)/len(montelist)
        montel.append(sum(montelist)/len(montelist))
        iteravg.append(it)
    print sum(montel)/len(montel), sum(iteravg)/len(iteravg)
    
