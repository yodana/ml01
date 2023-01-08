import numpy as np

def minmax(x):
    if (isinstance(x, np.ndarray) == True):
        x = np.squeeze(x)
        if (x.ndim == 1 and len(x) != 0):
            new = []
            if (np.max(x) - np.min(x) != 0):
                for t in x:
                    new.append((float(t - np.min(x)) ) / (float(np.max(x) - np.min(x))))
            return np.array(new)
    return None

if __name__ == '__main__':

    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print(minmax(X))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y))