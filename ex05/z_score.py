import numpy as np

def zscore(x):
    if (isinstance(x, np.ndarray) == True):
        x = np.squeeze(x)
        if (x.ndim == 1 and len(x) != 0):
            if (np.std(x) != 0)
                X = (x-np.mean(x)) / np.std(x)
                return X    
    return None

if __name__ == '__main__':

    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(zscore(Y))