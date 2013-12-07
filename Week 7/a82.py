import numpy as np
import scipy as sp

def getRandomLine():
    return zip(np.random.uniform(-1,1.00,2),np.random.uniform(-1,1.00,2))

def getPoints(numberOfPoints):
    pointList = zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints))
    return pointList

def isLeft(a, b, c):
	return 1 if ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0 else -1;


if __name__ == "__main__":
    nopoints = 10
    cluster = getPoints(nopoints)
    line = getRandomLine()
    sample = np.array([(i[0], i[1], isLeft(line[0], line[1], i)) for i in cluster])
    
