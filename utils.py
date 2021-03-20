import numpy as np

def searchBestIndicator(Z,F,C):
    obj = np.zeros((C,1))
    for j in range(C):
        obj[j,0] = np.power(np.linalg.norm(Z-F[:,j]),2)
    index = np.argmin(obj)
    return index+1