import numpy as np

class MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    

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

    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    # Example 0.0:
    y_hat = lr1.predict_(x)
    # Output:
    print(y_hat)
    print(lr1.loss_elem_(y, y_hat))
    # Output:

    # Example 0.2:
    print(lr1.loss_(y, y_hat))
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    print(lr2.fit_(x, y))
    # Output:
    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(y_hat)
    # Output:
    # Example 1.2:
    print(lr2.loss_elem_(y, y_hat))
    # Output:
    # Example 1.3:
    print(lr2.loss_(y, y_hat))
    # Output:
