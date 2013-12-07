import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import scimath

if __name__ == "__main__":
    '''k = np.arange(-5,5, 0.001)
    xneg = np.arange(-5,-1, 0.01)
    xpos = np.arange(1,5, 0.01)
    x = xneg + xpos
    print x
    plt.plot(x, scimath.sqrt((x-1) * (x + 1)),'r')
    plt.plot(x, -scimath.sqrt((x-1) * (x + 1)),'r')
    #plt.plot(k, y)
    plt.ylim(-5,5)
    plt.show()'''
    wlist = []
    for l in range(1,2):
        x1 = np.arange(0, (l*2), 0.01)
        x2 = np.arange(-(l*2), 0, 0.01)
        pointspositive1 = [(i, scimath.sqrt((i-l) * (i + l)),-1) for i in x1 if i >= l]
        pointspositive2 = [(i, -scimath.sqrt((i-l) * (i + l)),-1) for i in x1 if i >= l]
        points0positive = [(i, 0, 1) for i in x1 if i < l]
        pointsneg1 = [(i, scimath.sqrt((i-l) * (i + l)), -1) for i in x2 if i <= -l]
        pointsneg2 = [(i, -scimath.sqrt((i-l) * (i + l)), -1) for i in x2 if i <= -l]
        points0neg = [(i, 0, 1) for i in x2 if i > -l]
        
        pointsU = pointspositive1 + pointspositive2 + points0positive + pointsneg1 + pointsneg2 + points0neg
        x1list, x2list, ylist = zip(*pointsU)
        listoflist = [[1,x1list[i]**2, x2list[i]**2] for i in range(len(x1list))]
        listofY = [[i] for i in ylist]
        w = np.linalg.lstsq(listoflist, listofY)[0]
        wlist.append(w)
        plt.plot(x1list, x2list, "o")
    plt.show()
    for k in wlist:
        print k
