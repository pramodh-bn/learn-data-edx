#import Math as math 
import math

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

def equation1(x):
	return 1 + x

def equation2(x):
	return 1 + x + choose(x, 2)

def equation3(x):
	total = 0
	for i in range(math.floor(math.sqrt(x))):
		total += choose(x, i)
	return total

def equation4(x):
	return 2 ** math.floor(x/2)

def equation5(x):
	return 2 ** x

def comb1(x):
    return choose(x+1,4)

def comb2(x):
    return choose(x+1, 2) + 1

def comb3(x):
    return choose(x+1,4) + choose(x+1, 2) + 1

def comb4(x):
    return choose(x+1, 4) + choose(x+1, 3) + choose(x+1, 2) + choose(x+1, 1) + 1

def doEquation(m, nl):
    return [ 2 * m * math.exp(-2*0.05*0.05 * i) for i in nl]





if __name__ == '__main__':
    print('hello')
    i=10
    #for i in range(1, 10):
        #print("for ", i, " ", 2 ** i)
        #print(equation1(i))
        #print(equation2(i))
        #print(equation3(i))
        #print(equation4(i))
        #print(equation5(i))
        #print("************")
    print("i   2^n   c1   c2    c3    c4")
    for i in range(1, 7):
        print(i, "  ", 2**i, "  ", comb1(i), "  ", comb2(i), "  ", comb3(i), "  ", comb4(i))
    print(choose(4,3))

    mlist = [1, 10, 100]
    nlist = [500, 1000, 1500, 2000]
    k = [doEquation(i, nlist) for i in mlist]
    for j in k:
        print(j)

