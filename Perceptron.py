import numpy as np
class Perceptron:
    def __init__(self, lr=0.01, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights[1:] += update*xi
                self.weights[0] += update
                errors += int(update!=0.0)
            self.errors.append(errors)
        return self
    
    # Able to predict a matrix X -> y vector,
    # or predict a single xi -> yi scalar
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
      
    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
def test():
    X = np.array([[2.0, 1.0], [3.0, 4.0], [4.0, 2.0], [3.0, 1.0]])
    y = np.array([-1, 1, 1, -1])
    perceptron = Perceptron()
    # during training, only pass each vector xi to predict()
    perceptron.fit(X, y)

    new_X = np.array([[5.0, 2.0], [1.0, 3.0]])
    # pass a matrix X to predict()
    print(perceptron.predict(new_X))
    #array([-1,  1])

if __name__ == '__main__':
    test()
