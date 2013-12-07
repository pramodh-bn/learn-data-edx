import numpy as np
import scipy.spatial as sp

def errorFunc(u, v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2
    
def partial(u, v):
    gradient_u = 2 * (np.exp(v)+2*v*np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))
    gradient_v = 2 * (u*np.exp(v)-2*v*np.exp(-u)) * (u * np.exp(v) - 2*np.exp(-u))
    return np.array([gradient_u, gradient_v])

def partialu(u,v):
    print u, v
    gradient_u = 2 * (np.exp(v)+2*v*np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))
    print gradient_u
    return gradient_u

def partialv(u, v):
    gradient_v = 2 * (u*np.exp(v)-2*v*np.exp(-u)) * (u * np.exp(v) - 2*np.exp(-u))
    return gradient_v
      
def euclediandistance(p1, p2):
    #return np.linalg.norm(np.array([p1,p2]))
    #return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return sp.distance.pdist(np.array([p1,p2]), "euclidean")

def ass56():
    w = np.array([1,1])
    eta = 0.1
    for i in range(10):
        w = w - np.multiply(eta, partial(w[0], w[1]))
    print w, errorFunc(w[0], w[1])
    points = [(1,1),(0.713,0.045),(0.016, 0.112),(-0.083,0.029), (0.045, 0.024)]
    for i in points:
        print(euclediandistance(w, np.array([i[0],i[1]])))
    

if __name__ == "__main__":
    w = np.array([1,1])
    eta = 0.1
    print "At start ", w, errorFunc(w[0], w[1])
    for i in range(15):
        w = (w[0] - eta * partialu(w[0], w[1]), w[1])
        print "step 1 ", w, "error ", errorFunc(w[0], w[1])
        w = (w[0], w[1] - eta * partialv(w[0], w[1]))
        print "step 2 ", w, "error ", errorFunc(w[0], w[1])
        
    print "in the end ", w, errorFunc(w[0], w[1])
    
    