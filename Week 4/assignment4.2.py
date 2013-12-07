import numpy as np
import matplotlib.pyplot as plt

def doLinearRegression(a, b):
    listoflist = [[a[0]],[b[0]]]
    listofX = [a[0],b[0]]
    listofY = [a[1], b[1]]
    matX = np.matrix(listoflist)
    matY = np.matrix(listofY)
    #print(matX)
    #print(matY)
    #print(((matX.T*matX).I*matX.T)*matY.T)
    #print("this is ", np.matrix(np.linalg.pinv(listoflist))*matY.T)
    return np.linalg.lstsq(listoflist, listofY)[0][0]
    #print(((matX.T * matX).I * matX.T) * matY)
    #print(((matX.T * matX).I * matX.T)*matY.T)
    #w = ((matX.T * matX).I * matX.T)*matY.T
    #print(w.T*matX)
    #print(matY)
    #return np.squeeze(np.asarray(w))

def doTheLinearMethod(a, b):
    #g=(x1*y1+x2*y2)/(x1*x1+x2*x2)
    return (a[0]*a[1] + b[0] * b[1])/(a[0]*a[0] + b[0]*b[0])

def doExperiment(n):
    x = np.linspace(-1, 1, n)
    print(x)
    y = np.sin(np.pi * x)
    print(y)
    slopelist = []
    for i in range(n):
        index = np.random.randint(0,n, 2)
        slopelist.append(doTheLinearMethod((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]])))
    print(sum(slopelist)/len(slopelist))

#testDatalist = list()
def doRegressionExperiment(n):
    x = np.linspace(-1, 1, n)
    #print(x)
    y = np.sin(np.pi * x)
    #print(y)
    slopelist = []
    for i in range(n):
        index = np.random.randint(0,n, 2)
        slopelist.append(doLinearRegression((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]])))
        #testDatalist = slopelist
    print(len(slopelist))
    return sum(slopelist)/len(slopelist), slopelist

def doMonteCarlo(ghyp, n):
    x_ind = np.linspace(-1, 1, n)
    y_ind = np.sin(np.pi * x_ind)
    y_hyph = [ghyp * i for i in x_ind]
    squarelist = [(y_hyph[i]-y_ind[i])**2 for i in range(len(y_ind))]
    return sum(squarelist)/len(squarelist)
    
def doMonteCarloVariance(ghyp, n, testDatalist):    
    x_ind = np.linspace(-1, 1, n)
    errorlist = []
    for i in x_ind:
        jlist=[(ghyp*i - j*i)**2 for j in testDatalist]
        #print(jlist)
        errorlist.append(sum(jlist)/len(jlist))
    return sum(errorlist)/len(errorlist)            

if __name__ == '__main__':
    #print "hello"
    #plt.plot(x,y)
    #plt.show()  
    # pick 2 random points x1, x2 
    #doExperiment(2000)
    '''n = 2000
    x = np.linspace(-1, 1, n)
    print(x)
    y = np.sin(np.pi * x)
    print(y)
    index = np.random.randint(0,n, 2)
    k = doLinearRegression((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]]))
    print(k)'''
    #print(k * x[index[0]] - y[index[0]])
    #print(k * x[index[1]] - y[index[1]])
    m = [doRegressionExperiment(50) for i in range(10)]
    result, testdata = zip(*m)
    #print(result)
    #print(testdata)
    '''hyph = sum(result)/len(result)
    #hyph = doRegressionExperiment(1000)
    print(hyph)
    #print(testDatalist)
    monti = [doMonteCarloVariance(hyph, 2000, testdata[0]+testdata[1]) for i in range(1)]
    print(sum(monti)/len(monti))'''
    
    