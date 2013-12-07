# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:24:03 2013

@author: pramodh
"""

import numpy as np
from sklearn import svm

def my_kernel(x, xhyph):
    print("x", x)
    print("xhyph", xhyph)
    print(np.dot(x.T, xhyph))
    return (1 + np.dot(x.T, xhyph))**2
    

if __name__ == '__main__':
    clf = svm.SVC(C=10**10, kernel="poly", coef0=1, gamma=1, degree=2, verbose=True)
    print(clf.kernel)
    in_data = np.array([[1,0,-1],[0,1,-1], [0,-1,-1], [-1,0,1], [0,2,1],[0,-2,1], [-2,0,1]])
    X = np.c_[in_data[:,0], in_data[:,1]]
    y = in_data[:,2]
    #print(X)
    #print(y)
    clf.fit(X,y)
    print(clf.support_vectors_)
    
    #print(clf.predict(X))
