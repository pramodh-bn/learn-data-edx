import random

#random.seed(30)
def isLeft(a, b, c):
	return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;

def monetcarlo(pointa, pointb, weight, nopoint):
	points = [(random.uniform(-1,1), random.uniform(-1,1)) for i in range(nopoint)]
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

def dotheJob(nopoints):
	#random.seed(30)
	(x1, y1) = (random.uniform(-1,1), random.uniform(-1,1))
	(x2, y2) = (random.uniform(-1,1), random.uniform(-1,1))
	#(x1, y1) = (0.4319251147283589, 0.9399066761152763)
	#(x2, y2) = (0.046821730985817434, -0.3062852586405018)
	#print((x1, y1))
	#print((x2,y2))
	cluster = [(random.uniform(-1,1), random.uniform(-1,1)) for i in range(nopoints)]
	#cluster = [(0.91734654387618297, -0.46416631664610053), (0.41431196942272219, -0.76993224188032561), (0.15269476267089144, -0.51205098180554121), (0.39069641697947577, 0.61553664788698237),(-0.98094872995220017, 0.74886780890877147), (-0.46280829377284949, 0.76514123910230047), (-0.17618456109624958, -0.60020141626620238), (-0.78662467306484385, 0.31435401008730968), (-0.75044363518378687, -0.065471192771189513), (-0.99659717449102314, -0.58381239070913815)]
	#print(cluster)
	positives = [i for i in cluster if isLeft((x1,y1),(x2,y2),i)]
	negatives = [i for i in cluster if not isLeft((x1,y1),(x2,y2),i)]
	#print(len(positives) + len(negatives))
	w = [0,0,0]
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
			choice = random.choice(mismatch)
			#print("choice ", choice)
			w = [w[i]+choice[i] for i in range(3)]
			#print("new weight ", w)
		else:
			break
	#print("this is the iteration ", iteration)
	#print("this is the weight ", w)
	montelist = [monetcarlo((x1,y1),(x2,y2),w,1000) for i in range(5)]
	#print("Montelist " , montelist)
	monteavg = sum(montelist)/len(montelist)
	return (iteration, monteavg)



if __name__ == '__main__':
	avgofavgiters = list()
	avgofavgprob = list()
	nopoints = 300
	for a in range(20):
		k = [dotheJob(nopoints) for i in range(10)]
		iteravg = sum([i[0] for i in k])/len(k)
		probavg = sum([i[1] for i in k])/len(k)
		#print("avg. iteration ", iteravg, " avg. prob ", probavg)
		avgofavgiters.append(iteravg)
		avgofavgprob.append(probavg)
	print(" avg of avg iters for 100 ", sum([i for i in avgofavgiters])/len(avgofavgiters))
	print(" avg of avg prob for 100 ", sum([i for i in avgofavgprob])/len(avgofavgprob))
