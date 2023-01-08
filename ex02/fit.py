import numpy as np
import matplotlib.pyplot as plt

def predict_(x, theta):
    if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True):
        x = np.squeeze(x)
        if (x.ndim == 1 and len(x) != 0 and theta.shape == (2,)):
            x = np.c_[ np.ones(len(x)) , x]
            y = x.dot(theta)
            return y
    return None

def gradient(x, y, theta):
    if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        theta = np.squeeze(theta)
        x = np.squeeze(x)
        y = np.squeeze(y)
        if (x.ndim == 1 and y.ndim == 1 and theta.shape == (2,) and len(x) == len(y) and len(x) != 0):
            xt = np.c_[ np.ones(len(x)) , x]
            r = (xt.T.dot(predict_(x, theta) - y)) / len(x)
            return r
    return None


def fit_(x, y, theta, alpha, max_iter):
     if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        theta = np.squeeze(theta)
        x = np.squeeze(x)
        y = np.squeeze(y)
        if (x.ndim == 1 and y.ndim == 1 and theta.shape == (2,) and len(x) == len(y) and len(x) != 0):
            while(max_iter != 0):
                g = gradient(x, y, theta)
                theta = theta - alpha*g
                max_iter = max_iter - 1
            return theta

if __name__ == '__main__':
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1]).reshape((-1, 1))
    # Example 0:
    #theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    theta1 = np.array([1.40709365,1.1150909 ])
    # Output
    print(predict_(x, theta1))