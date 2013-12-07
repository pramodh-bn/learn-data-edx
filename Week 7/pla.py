import time
 
import numpy as np
import matplotlib as mpl
import scipy as sp
import matplotlib.pyplot as plt
 
pointsTotal = 100
numTestPoints = 100000
experiments = 1000
 
averageError = 0
averageIterations = 0
for i in range(experiments):
 
## creating the line
 
	l1 = np.random.uniform(-1, 1, 2)
	l2 = np.random.uniform(-1, 1, 2)
 
	def myline(x):
		return l1[1] + (l2[1] - l1[1]) / (l2[0] - l1[0]) * ( x - (l1[0]))
 
	def targetFunction(point):
		if point[2] > myline(point[1]):
			return 1
		else:
			return -1
 
 
## creating the sampling points
 
	points = np.random.uniform(-1,1,(pointsTotal,2))
	zeros = np.ones( (pointsTotal,1) )
	points = np.append(zeros, points, axis = 1)
 
 
## fitting the w function
 
	w = np.zeros( 3 )
 
	iteration = 0
 
	done = False
	while not done:
		iteration += 1
		wrongpoints = 0
		for p in points:
			if np.sign( np.dot(w, p) ) != targetFunction( p ):
				w = np.add( w, targetFunction( p ) * p )
				wrongpoints += 1
				break
			if wrongpoints == 0:
				done = True
 
 
	x = np.array( [-1,1] )
 
 
## testing coverage
 
	testPoints = np.random.uniform(-1,1,(numTestPoints,2))
	zeros = np.ones( (numTestPoints,1) )
	testPoints = np.append(zeros, testPoints, axis = 1)
 
	testPointsWrong = 0
	for p in testPoints:
		if np.sign( np.dot(w, p) ) != targetFunction( p ):
			testPointsWrong += 1
 
	averageError += testPointsWrong / numTestPoints
	averageIterations += iteration
 
print( "exp: " + str(i) + " error: " + str(testPointsWrong / numTestPoints) + " averageError: " + str( averageError / (i+1) ) + " iterations: " + str(iteration) + " averageIterations: " + str( averageIterations / (i+1) ))

