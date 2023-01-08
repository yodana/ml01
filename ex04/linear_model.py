import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
class MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    
    def mse_(y, y_hat):
        j = 0
        if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            y = np.squeeze(y)
            y_hat = np.squeeze(y_hat)
            if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
                m = y_hat - y
                j = m.dot(m)
                return float(j / (len(y)))
        return None

    def gradient(self, x, y):
        if (isinstance(x, np.ndarray) == True or isinstance(self.thetas, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            theta = np.squeeze(self.thetas)
            x = np.squeeze(x)
            y = np.squeeze(y)
            if (x.ndim == 1 and y.ndim == 1 and self.thetas.shape == (2,) and len(x) == len(y) and len(x) != 0):
                xt = np.c_[ np.ones(len(x)) , x]
                r = (xt.T.dot(self.predict_(x) - y)) / len(x)
                return r
        return None

    def fit_(self, x, y):
     if (isinstance(x, np.ndarray) == True or isinstance(self.thetas, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        self.thetas = np.squeeze(self.thetas)
        x = np.squeeze(x)
        y = np.squeeze(y)
        if (x.ndim == 1 and y.ndim == 1 and self.thetas.shape == (2,) and len(x) == len(y) and len(x) != 0):
            while(self.max_iter != 0):
                g = self.gradient(x, y)
                self.thetas = self.thetas - self.alpha*g
                self.max_iter = self.max_iter - 1
            return self.thetas
    
    def predict_(self, x):
        if (isinstance(x, np.ndarray) == True or isinstance(self.thetas, np.ndarray) == True):
            x = np.squeeze(x)
            self.thetas = np.squeeze(self.thetas)
            if (x.ndim == 1 and len(x) != 0 and self.thetas.shape == (2,)):
                x = np.c_[ np.ones(len(x)) , x]
                y = x.dot(self.thetas)
                return y
        return None
    
    def loss_elem_(self, y, y_hat):
        j = []
        if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
                for i in range(0, len(y)):
                    j.append((y_hat[i] - y[i])**2)
                return np.array(j)
        return None

    def loss_(self, y, y_hat):
        j = 0
        if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            y = np.squeeze(y)
            y_hat = np.squeeze(y_hat)
            if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
                m = y_hat - y
                j = m.dot(m)
                return float(j / (2*len(y)))
        return None

if __name__ == '__main__':
    data = pd.read_csv("are_blue_pills_magic.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1,1)
    linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
    linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
    Y_model2 = linear_model2.predict_(Xpill)
    #first 
    Y_model1 = linear_model1.predict_(Xpill)
    plt.scatter(Xpill, Yscore)
    plt.scatter(Xpill, Y_model1, color="green")
    plt.plot(Xpill, Y_model1, "--", color="green")
    plt.xlabel('Space driving score')
    plt.ylabel('Quantity of blue pill (in micrograms)')
    plt.show()
    
    #second
    myMap = plt.get_cmap('Greys')
    theta1 = np.linspace(-17, -3, 100)
    theta0 = np.array([89.0, 92, 95, 97, 99, 102])
    l = []
    legend = []
    c = 0.99
    hex = 0xDCDCDC
    for t0 in theta0:
        for t1 in theta1:
            y_pred = MyLinearRegression(np.array([[t0], [t1]])).predict_(Xpill)
            l.append( MyLinearRegression(np.array([[t0], [t1]])).loss_(Yscore, y_pred))
        color = myMap(c)
        plt.plot(theta1, l, color=color)
        plt.ylabel('Cost function J(θ0, θ1)')
        plt.xlabel('θ1')
        legend.append("J(θ0=" + str(t0) + ",θ1)")
        l = []
        hex = hex - 20
        c = c - 0.1
    plt.legend(legend)
    plt.show()
    
    #third 
    print(MyLinearRegression.mse_(Yscore, Y_model2))
    print(mean_squared_error(Yscore, Y_model1))
    print(MyLinearRegression.mse_(Yscore, Y_model2))
    print(mean_squared_error(Yscore, Y_model2))
