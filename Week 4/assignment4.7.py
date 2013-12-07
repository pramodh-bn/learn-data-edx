import numpy as np

def doLinearRegressionA(a, b):
    listoflist = [[1],[1]]
    listofY = [a[1], b[1]]
    return np.linalg.lstsq(listoflist, listofY)[0]

def doLinearRegressionB(a, b):
    listoflist = [[a[0]],[b[0]]]
    listofY = [a[1], b[1]]
    return np.linalg.lstsq(listoflist, listofY)[0]

def doLinearRegressionC(a, b):
    listoflist = [[1, a[0]],[1, b[0]]]
    listofY = [a[1], b[1]]
    return np.linalg.lstsq(listoflist, listofY)[0]

def doLinearRegressionD(a, b):
    listoflist = [[a[0]**2],[b[0]**2]]
    listofY = [a[1], b[1]]
    return np.linalg.lstsq(listoflist, listofY)[0]

def doLinearRegressionE(a, b):
    listoflist = [[1,a[0]**2],[1, b[0]**2]]
    listofY = [a[1], b[1]]
    return np.linalg.lstsq(listoflist, listofY)[0]

def monteA(eq, x, y):
    a = [(eq[0]-y[i])**2 for i in range(len(x))]
    return sum(a)/len(a)

def monteB(eq, x, y):
    b = [(eq[0]*x[i]-y[i])**2 for i in range(len(x))]
    return sum(b)/len(b)

def monteC(eq, x, y):
    c = [((eq[0] + eq[1]*x[i])-y[i])**2 for i in range(len(x))]
    return sum(c)/len(c)

def monteD(eq, x, y):
    d = [((eq[0]*x[i]*x[i])-y[i])**2 for i in range(len(x))]
    print(sum(d))
    return sum(d)/len(d)

def monteE(eq, x, y):
    e = [((eq[0] + eq[1]*x[i]*x[i])-y[i])**2 for i in range(len(x))]
    return sum(e)/len(e)

if __name__ == "__main__":
    print "hello"
    n = 1000
    x = np.linspace(-1,1,n)
    y = np.sin(np.pi*x)
    glist = []
    montelist = []
    errd = []
    erre = []
    for i in range(n):
        index = np.random.randint(0,n, 2)
        a = doLinearRegressionA((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]]))
        print("a ",a)
        b = doLinearRegressionB((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]]))
        print("b", b)
        c = doLinearRegressionC((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]]))
        print("c", c)
        d = doLinearRegressionD((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]]))
        print("d", d)
        e = doLinearRegressionE((x[index[0]], y[index[0]]), (x[index[1]], y[index[1]]))
        print("e", e)
        glist.append((a, b, c, d, e))
        error_a = monteA(a, x, y)
        print("error_A ", error_a)
        error_b = monteB(b, x, y)
        print("error_B ", error_b)
        error_c = monteC(c, x, y)
        print("error_C ",error_c)
        error_d = monteD(d, x, y)
        print("error_D ",error_d)
        error_e = monteE(e, x, y)
        print("error E ",error_e)
        montelist.append((error_a,error_b,error_c,error_d,error_e))
        errd.append(error_d)
        erre.append(error_e)
    #print(glist)    
    monte_a, monte_b, monte_c, monte_d, monte_e = zip(*(montelist))
    print("a", sum(monte_a)/len(monte_a))
    print("b", sum(monte_b)/len(monte_b))
    print("c", sum(monte_c)/len(monte_c))
    print("d", sum(monte_d)/len(monte_d))
    print("d", sum(errd)/len(errd))
    print("e", sum(monte_e)/len(monte_e))
    print("e", sum(erre)/len(erre))
    
    