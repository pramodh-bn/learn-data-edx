import numpy as np

def isLeft(a, b, c):
	return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;

def monetcarlo(pointa, pointb, weight, nopoint):
    #print "weights ", weight
    points = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoint)]
    positives = [i for i in points if isLeft(pointa,pointb,i)]
    negatives = [i for i in points if not isLeft(pointa, pointb,i)]
    mismatches = 0
    for i in points:
	yy = weight[0] + weight[1] * i[0] + weight[2] * i[1]
	if(yy>0):
           if(i not in positives):
           	mismatches = mismatches + 1
        else:
	   if(i not in negatives):
		mismatches = mismatches + 1
    #print("mismatches ", mismatches)
    return float(mismatches)/nopoint

def doPLA(positives, negatives, weight):
	#random.seed(30)
	#(x1, y1) = (np.random.uniform(-1,1), np.random.uniform(-1,1))
	#(x2, y2) = (np.random.uniform(-1,1), np.random.uniform(-1,1))
	#(x1, y1) = (0.4319251147283589, 0.9399066761152763)
	#(x2, y2) = (0.046821730985817434, -0.3062852586405018)
	#print((x1, y1))
	#print((x2,y2))
	#cluster = [(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(nopoints)]
	#cluster = [(0.91734654387618297, -0.46416631664610053), (0.41431196942272219, -0.76993224188032561), (0.15269476267089144, -0.51205098180554121), (0.39069641697947577, 0.61553664788698237),(-0.98094872995220017, 0.74886780890877147), (-0.46280829377284949, 0.76514123910230047), (-0.17618456109624958, -0.60020141626620238), (-0.78662467306484385, 0.31435401008730968), (-0.75044363518378687, -0.065471192771189513), (-0.99659717449102314, -0.58381239070913815)]
	#print(cluster)
	#positives = [i for i in cluster if isLeft((x1,y1),(x2,y2),i)]
	#negatives = [i for i in cluster if not isLeft((x1,y1),(x2,y2),i)]
	#print(len(positives) + len(negatives))
	w = weight
	cluster = positives + negatives
	iteration = 0
	it = 0
	while True:#(it < 10):
		iteration = iteration + 1
		it = it + 1
		mismatch = list()
		for i in cluster:
			#print("point in question ", i , " weight ", w)
			yy = w[0] + w[1] * i[0] + w[2] * i[1]
			#print("this is after applying weight to a point ",yy)
			if(yy>0):
				if(i not in positives):
					#print(i, " is not in positives ", positives)
					mismatch.append((-1, -i[0],-i[1]))
			else:
				if(i not in negatives):
					#print(i, " is not in negatives ", negatives)
					mismatch.append((1, (i[0]), (i[1])))
		#print(" length ", len(mismatch), " mismatch list ",mismatch)
		if(len(mismatch) > 0):
			#find a random point and update w
			choiceIndex = np.random.randint(0, len(mismatch))
			choice = mismatch[choiceIndex]
			#print("choice ", choice)
			w = [w[i]+choice[i] for i in range(3)]
			#print("new weight ", w)
		else:
			break
	#print("this is the iteration ", iteration)
	#print("this is the weight ", w)
	#montelist = [monetcarlo((x1,y1),(x2,y2),w,10000) for i in range(5)]
	#print("Montelist " , montelist)
	#monteavg = sum([i for i in montelist])/10
	return iteration

def getRandomLine():
    return zip(np.random.uniform(-1,1.00,2),np.random.uniform(-1,1.00,2))

def getPoints(numberOfPoints):
    pointList = zip(np.random.uniform(-1,1.00,numberOfPoints),np.random.uniform(-1,1.00,numberOfPoints))
    return pointList

def regressionAlgo(positives, negatives):
    #print(line)
    #cluster = getPoints(numberOfPoints)
    #print "positive", positives, len(positives)
    #print "negatives", negatives, len(negatives) 
    totallist = positives + negatives
    #print "total ", totallist, len(totallist)
    listoflist = [[1, i[0], i[1]] for i in totallist]
    listofY = [[1] for i in range(len(positives))] + [[-1] for i in range(len(negatives))]
    matX = np.matrix(listoflist)
    matY = np.matrix(listofY)
    #print matX
    #print matY
    return ((matX.T * matX).I * matX.T) * matY

def getMismathces(positives, negatives, weights):
    #print weights
    mismatches = 0
    allinlist = positives + negatives
    #print(allinlist)
    for i in allinlist:
        xlist = [1, i[0], i[1]]
        #print "xlist ", xlist
        yy = np.squeeze(np.asarray(weights)).dot(xlist)
        if (yy > 0 and i not in positives) or (yy < 0 and i not in negatives):
            mismatches += 1
        #print "yy is ", yy
    return mismatches

def part1():
	avgofmissed = list()
	montecarloprob = list()
	nopoints = 100
	noExperiments = 1000
	for i in range(noExperiments):
       	    cluster = getPoints(nopoints)
       	    line = getRandomLine()
            positives = [i for i in cluster if isLeft(line[0],line[1],i)]
            negatives = [i for i in cluster if not isLeft(line[0],line[1],i)]
       	    weights = regressionAlgo(positives, negatives)
       	    #print weights
       	    missed = getMismathces(positives, negatives, weights)
       	    #print("this is missed", missed)
       	    avgofmissed.append(float(missed)/nopoints)
       	    montevals = [monetcarlo(line[0], line[1], np.squeeze(np.asarray(weights)), 1000) for i in range(1)]
       	    montecarloprob.append(sum(montevals))
	print "avgofmissed ", avgofmissed
	print "monte carlo ", montecarloprob
	print "avgof ", sum(avgofmissed)/noExperiments
	print "avgof monte carlo ", sum(montecarloprob)/noExperiments
    

if __name__ == '__main__':
	avgofmissed = list()
	montecarloprob = list()
	iterations = list()
	nopoints = 100
	noExperiments = 2000
	for i in range(noExperiments):
       	    cluster = getPoints(nopoints)
       	    line = getRandomLine()
            positives = [i for i in cluster if isLeft(line[0],line[1],i)]
            negatives = [i for i in cluster if not isLeft(line[0],line[1],i)]
       	    weights = regressionAlgo(positives, negatives)
       	    it = doPLA(positives, negatives, np.squeeze(np.asarray(weights)))
       	    #print "iterations ", it
       	    iterations.append(it)
       	#print iterations
       	print "avg iterations= ", float(sum(iterations))/noExperiments
       	    #print weights
       	    #missed = getMismathces(positives, negatives, weights)
       	    #print("this is missed", missed)
       	    #avgofmissed.append(float(missed)/nopoints)
       	    #montevals = [monetcarlo(line[0], line[1], np.squeeze(np.asarray(weights)), 1000) for i in range(1)]
       	    #montecarloprob.append(sum(montevals))
	#print "avgofmissed ", avgofmissed
	#print "monte carlo ", montecarloprob
	#print "avgof ", sum(avgofmissed)/noExperiments
	#print "avgof monte carlo ", sum(montecarloprob)/noExperiments
