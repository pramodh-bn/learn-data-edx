# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:13:49 2013

@author: dyanna
"""

import numpy as np
import scipy as sp
from sklearn.svm import SVC
from sklearn import cross_validation


def classifyPoints(num, sample, vs=-1.0):
    if vs == -1.0:
        # vs all
        negatives = sample[sample[:,0] != num]
    else:
        negatives = sample[sample[:,0] == vs]

    positives = sample[sample[:,0] == num]
    listpositive = np.empty(len(positives))
    listpositive.fill(1.0)
    datapositive = np.c_[positives[:,1],positives[:,2], listpositive]
    negativelist = np.empty(len(negatives))
    negativelist.fill(-1.0)
    datanegative = np.c_[negatives[:,1], negatives[:,2], negativelist]
    newsample = np.concatenate((datanegative, datapositive), axis=0)
    np.random.shuffle(newsample)
    return newsample


if __name__ == "__main__":
    in_data = np.genfromtxt("features.train", dtype=float)
    out_data = np.genfromtxt("features.test", dtype=float)
    clist = [0.0001, 0.001, 0.01, 0.1, 1.0]
    num = 1.0
    vs = 5.0
    c = 0.001
    clf = SVC(C = c, kernel = 'rbf', gamma=1.0)  
    sample = classifyPoints(num, in_data, vs)
    X = np.c_[sample[:,0], sample[:,1]]
    y = sample[:,2]
    '''kf = cross_validation.KFold(len(y), n_folds=10, indices=False)
    print(kf)
    for train, test in kf:
        print("%s %s" % (train, test))'''
    ss = cross_validation.ShuffleSplit(len(y), test_size=0.1, random_state=0, indices=False)
    print(ss)
    print(len(ss))
    for i in clist:
        clf = SVC(C = i, kernel = 'poly', degree=2, coef0=1, gamma=1.0)  
        eins = 1-cross_validation.cross_val_score(clf, X, y, cv=ss)
        print("%s %s" %(i, np.mean(eins)))
