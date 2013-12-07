import numpy as np

if __name__ == "__main__":
    e1list = list()
    e2list = list()
    elist = list()
    for i in range(10000):
        e1 = np.random.uniform(size=100000)
        e2 = np.random.uniform(size=100000)
        e = np.minimum(e1, e2)
        e1list.append(np.average(e1))
        e2list.append(np.average(e2))
        elist.append(np.average(e))
    print sum(e1list)/len(e1list), sum(e2list)/len(e2list), sum(elist)/len(elist)
    