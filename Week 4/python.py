import matplotlib.pyplot as plt
import numpy as np
import math

def power(n, dimension):
	if(n>dimension):
		return math.pow(n, dimension)
	else:
		return math.pow(2,n)

def mH(n, dimension):
	if(n > dimension):
		return n, dimension
	else:
		return 2, n 

def original(n, delta, dimension):
	values = list()
	for i in range(n):
		mhVal = mH(2*(i+1), dimension)
		#l = np.sqrt((8.0/(i+1)) * np.log((4*(power(2*(i+1),dimension)))/delta))
		logpart = np.log(4) + mhVal[1] * np.log(mhVal[0]) - np.log(delta)
		l = np.sqrt((8.0/(i+1)) * logpart)
		values.append((i+1,l))
	return values

def rademacher(n, delta, dimension):
	values = []
	for i in range(n):
		l1 = np.sqrt((2*(np.log(2 * (i+1) * power(i+1, dimension))))/(i+1))
		l2 = np.sqrt((2.0/(i+1))*np.log(1.0/delta))
		l3 = 1.0/(i+1)
		l = l1+l2+l3
		values.append((i+1,l))
	return values

def parrondo(n, delta, dimension):
	values = []
	eps = 0.001
	for i in range(n):
		mhVal = mH(2*(i+1), dimension)
		logpart = np.log(6) + mhVal[1] * np.log(mhVal[0]) - np.log(delta)
		eps = np.sqrt((1.0/(i+1)) * ((2*eps) + logpart))
		values.append((i+1,eps))
	return values

def devroye(n, delta, dimension):
	values = []
	eps = 0.001
	for i in range(n):
		mhVal = mH((i+1)*(i+1), dimension)
		logpart = np.log(4) + mhVal[1] * np.log(mhVal[0]) - np.log(delta)
		print(eps, logpart)
		eps = np.sqrt((1.0/(2*(i+1))) * ((4 * eps * (1+eps)) + logpart))
		values.append((i+1,eps))
	return values

#plt.plot(zip(*original(10000, 0.05, 50)), marker="o", c="red")
#plt.plot(zip(*rademacher(10000, 0.05, 50)), marker="o", c="blue")
#plt.plot(zip(*parrondo(10000, 0.05, 50)), marker="o", c="green")
#plt.plot(zip(*devroye(10000, 0.05, 50)), marker="+", c="yellow")
n = 5
x, y = zip(*original(n, 0.05, 50))
#print(x, y)
plt.plot(x, y, 'r', label="Original")
x1, y1 = zip(*rademacher(n, 0.05, 50))
plt.plot(x1, y1, 'b', label="Rademacher")
#print(x1, y1)
x2, y2 = zip(*parrondo(n, 0.05, 50))
plt.plot(x2, y2, 'g', label="Parrondo")
x3, y3 = zip(*devroye(n, 0.05, 50))
plt.plot(x3, y3,'y', label="Devroye")
plt.legend(loc='upper right')
plt.xlabel("iteration")
plt.ylabel("epsilon")
plt.show()
print(y[n-1])
print(y1[n-1])
print(y2[n-1])
print(y3[n-1])