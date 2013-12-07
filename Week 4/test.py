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
		print(n)
		return 2, n


def original(n, delta, dimension):
	values = list()
	for i in range(n):
		l = math.sqrt((8.0/(i+1)) * math.log((4*(power(2*(i+1),dimension)))/delta))
		values.append((i+1,l))
	return values

def rademacher(n, delta, dimension):
	values = []
	for i in range(n):
		l1 = math.sqrt((2*(math.log(2 * (i+1) * power(i+1, dimension))))/(i+1))
		l2 = math.sqrt((2.0/(i+1))*(1.0/delta))
		l3 = 1.0/(i+1)
		l = l1+l2+l3
		values.append((i+1,l))
	return values

def parrondo(n, delta, dimension):
	values = []
	eps = 0.00001
	for i in range(n):
		eps = math.sqrt((1.0/(i+1)) * ((2*eps) + math.log((6 * power(2*(i+1), dimension))/delta)))
		values.append((i+1,eps))
	return values

def devroye(n, delta, dimension):
	values = []
	eps = 0.00001
	for i in range(n):
		mhVal = mH((i+1)*(i+1),dimension)
		second = math.log(4) + mhVal[1] * math.log(mhVal[0]) - math.log(delta)
		print(mhVal, " ", second)
		eps = math.sqrt((1.0/(2*(i+1))) * ((4 * eps * (1+eps)) + second))
		values.append((i+1,eps))
	return values

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def recform(n, q):
	print(n)
	if(n==1):
		return 2
	else:
		return 2 * recform(n-1,q) - choose(n-1, q)


if __name__ == '__main__':
	'''NList = [400000, 420000, 440000, 460000, 480000]
	N = 400000
	delta = 0.05
	mh = (2 * N) ** 10
	coeff = 8.0/N
	logval = (4 * mh)/delta
	logcoeff = math.log(logval)
	eps = math.sqrt(coeff * logcoeff)
	print(eps)
	k = [math.sqrt((8.0/i) * math.log((4*((2*i)**10))/delta)) for i in NList]
	print(k)
	k = [math.fabs(i-0.05) for i in k]
	print(k)
	m,k = zip(*original(5, 0.05, 10))
	print(original(5, 0.05, 10))
	print(rademacher(5, 0.05, 10))
	print(parrondo(5, 0.05, 10))
	print(devroye(5, 0.05, 10))
	a = mH(5,2)
	print(a[0], a[1], math.pow(a[0], a[1]))
	print(devroye(5, 0.05, 50))'''
	n = 10
	q = 10
	m = []
	for i in range(2, 11):
		m.append(recform(i, q))
	print(m)


