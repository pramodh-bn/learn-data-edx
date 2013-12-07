import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #print "hello"
    x = np.linspace(-1,1,1000)
    y = np.sin(np.pi*x)
    plt.ylim(-5,5)
    plt.xlim(-1,1)
    plt.plot(x,y)
    avglist = []
    for k in range(20):
        index = np.random.randint(0,1000, 2)
        listoflist = [[1,0,x[index[0]]**2],[1,0,x[index[1]]**2]]
        #listoflist = [[1],[1]]
        listofY = [y[index[0]], y[index[1]]]
        w = np.linalg.lstsq(listoflist, listofY)[0]
        print(w)
        ydash = [w[0]+w[2]*i*i for i in x]
        plt.plot(x,ydash,"r")
        errorlist = [(w[0]+w[2]*i*i - (np.sin(np.pi*i)))**2 for i in np.linspace(-1,1,1000)]
        avg = sum(errorlist)/len(errorlist)
        print("avg error", avg)
        avglist.append(avg)
    print(sum(avglist)/len(avglist))
    plt.show()  
    
