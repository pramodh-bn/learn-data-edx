import numpy as np

if __name__ == "__main__":
    arr = np.array([[1,2,1], [1,-2,-1], [3,2,0]])
    point = [1,2,-1]
    print arr[:]
    print point[:]
    print any(np.equal(arr, point).all(1))
