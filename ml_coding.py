import numpy as np

class LinearRegression:
    def __init__(self):
        self.m = None
        self.b = None
    
    def fit(self, X, y):
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = 0
        denominator = 0
        for i in range(n):
            numerator+=(X[i] - x_mean)*(y[i]-x_mean)
            denominator+=(X[i]-x_mean)**2
        self.m = numerator/denominator
        self.b = y_mean - self.m*x_mean

    def predict(self, X):
        y = []
        for i in range(len(X)):
            y_pred = self.m*X[i]+self.b
            y.append(np.round(y_pred, 1))
        return y


def test_linear_reg():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    lr = LinearRegression()
    lr.fit(X, y)
    print(lr.m)  # Output: 0.6
    print(lr.b)  # Output: 2.2
    y_pred = lr.predict(X)
    print(y_pred)  # Output: [2.8, 3.4, 4.0, 4.6, 5.2]


class LogisticRegression:
    def __init__(self, learning_rate = 0.01, n_iter = 10):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for i in range(self.n_iter):
            # cost function
            z = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(z)
            cost = -(1/n_samples) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
            # calculate gradient
            dw = 1/n_samples * np.dot(X.T, (y_pred - y))
            db = 1/n_samples * np.sum(y_pred - y)
            # Gradient Descent
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(z)
        # convert to binary prediction
        return np.round(y_pred).astype(int)
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
def test_logistic_reg():
    # train
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression()
    lr.fit(X, y)
    # inference
    X_new = np.array([[6, 7], [7, 8]])
    y_pred = lr.predict(X_new)
    print(y_pred)  # [1, 1]

import matplotlib.pyplot as plt
def visualize_log_reg():
    # create 2D dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # initialize logistic regression model
    lr = LogisticRegression()

    # train model on dataset
    lr.fit(X, y)

    # plot decision boundary
    x1 = np.linspace(0, 6, 100)
    x2 = np.linspace(0, 8, 100)
    xx, yy = np.meshgrid(x1, x2)
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plot data points
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)

    plt.show()


if __name__ == '__main__':
    while True:
        i = input("Which test?(Type quit to end) 1.Linear Reg 2.Logistic Reg\n")
        if i == '1':
            test_linear_reg()
        elif i == '2':
            test_logistic_reg()
            visualize_log_reg()
        elif i == 'quit':
            quit()
        else:
            print('Please choose a number')
