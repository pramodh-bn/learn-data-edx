# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:54:02 2013

@author: pramodh
"""

import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    in_data = np.array([[1,0,-1],[0,1,-1], [0,-1,-1], [-1,0,1], [0,2,1],[0,-2,1], [-2,0,1]])
    print(in_data)
    trans_data = np.c_[in_data[:,1]**2-2*in_data[:,0]-1,in_data[:,0]**2-2*in_data[:,1]+1,in_data[:,2]]
    print(trans_data)
    #plt.plot(trans_data[:,0], trans_data[:,1], "o")
    plus_data = trans_data[trans_data[:,2] == 1]
    minus_data = trans_data[trans_data[:,2] == -1]
    x = np.array( [-5,5] )
    w = [0.5, 1, -1]
    #plt.plot( x, -w[1]/w[2] * x - w[0] / w[2] , 'r' ) # this will throw an error if w[2] == 0
    list1 = np.empty(len(x))
    list1.fill(0)
    plt.plot( x, list1 , 'r' )
    plt.plot(plus_data[:,0], plus_data[:,1], "ro")
    plt.plot(minus_data[:,0], minus_data[:,1], "bs")
    plt.show()
    

