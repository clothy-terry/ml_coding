import numpy as np

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
        # cost function
        for i in range(self.n_iter):
            z = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(z)
            cost = -(1/n_samples) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))\
            
            # calculate gradient
            dw = 1/n_samples * np.dot(X.T, (y_pred - y))
            db = 1/n_samples * np.sum(y_pred - y)

            # Gradient Descent
            w -= self.learning_rate*dw
            b -= self.learning_rate*db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(z)
        # convert to binary prediction
        return np.round(y_pred).astype(int)

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
def test():
    # create sample dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # initialize logistic regression model
    lr = LogisticRegression()

    # train model on sample dataset
    lr.fit(X, y)

    # make predictions on new data
    X_new = np.array([[6, 7], [7, 8]])
    y_pred = lr.predict(X_new)

    print(y_pred)  # [1, 1]

if __name__ == '__main__':
    test()