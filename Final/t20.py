# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:35:20 2013

@author: pramodh
"""

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
    
    #g1    
    g1_index = np.random.randint(0,1000, 2)
    g1_listoflist = [[1,x[g1_index[0]]],[1,x[g1_index[1]]]]
        #listoflist = [[1],[1]]
    g1_listofY = [y[g1_index[0]], y[g1_index[1]]]
    g1_w = np.linalg.lstsq(g1_listoflist, g1_listofY)[0]
    print(g1_w)
    g1_ydash = [g1_w[0]+g1_w[1]*i for i in x]
    plt.plot(x,g1_ydash,"r")

    #g2
    g2_index = np.random.randint(0,1000, 2)
    g2_listoflist = [[1,x[g2_index[0]]],[1,x[g2_index[1]]]]
        #listoflist = [[1],[1]]
    g2_listofY = [y[g2_index[0]], y[g2_index[1]]]
    g2_w = np.linalg.lstsq(g2_listoflist, g2_listofY)[0]
    print(g2_w)
    g2_ydash = [g2_w[0]+g2_w[1]*i for i in x]
    plt.plot(x,g2_ydash,"g")
    
    for j in range(10):
        g1_errorlist = []
        g2_errorlist = []
        g_errorlist = []
        for i in np.random.uniform(-1,1,1000):
            g1_errorlist.append((g1_w[0]+g1_w[1]*i - (np.sin(np.pi*i)))**2)
            g2_errorlist.append((g2_w[0]+g2_w[1]*i - (np.sin(np.pi*i)))**2)
            g = 0.5 * ((g1_w[0]+g1_w[1]*i) + (g2_w[0]+g2_w[1]*i))
            g_errorlist.append((g - (np.sin(np.pi*i)))**2)
        print(np.mean(g1_errorlist) , np.mean(g2_errorlist), np.mean(g_errorlist), (np.mean(g1_errorlist)+np.mean(g2_errorlist))/2)    
    
    #g1_errorlist = [(w[0]+w[1]*i - (np.sin(np.pi*i)))**2 for i in np.linspace(-1,1,1000)]
    #avg = sum(errorlist)/len(errorlist)
    plt.show()  
