import numpy as np
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.p = {}
        self.p['W1'] = np.random.randn(input_size, hidden_size)
        self.p['b1'] = np.zeros(hidden_size)
        self.p['W2'] = np.random.randn(hidden_size, output_size)
        self.p['b2'] = np.zeros(output_size)

    def loss(self, X, y):
        probs = self.forward(X)
        correct_logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(correct_logprobs)
        return 1.0/len(X) * data_loss


    def forward(self, X):
        W1, b1 = self.p['W1'], self.p['b1']
        W2, b2 = self.p['W2'], self.p['b2']
        z1 = np.dot(X, W1) + b1
        h1 = np.maximum(0, z1)
        z2 = np.dot(h1, W2) + b2
        exp_z2 = np.exp(z2)
        Output = exp_z2/np.sum(exp_z2, axis = 1, keepdims=True)
        #Keep the dimension of exp_z2 after sum, so that / won't cause error.
        return Output
    
    def train(self, X, y, num_epochs, learning_rate = 0.1):
        for epoch in range(num_epochs):
            # Forward propagation
            z1 = np.dot(X, self.p['W1']) + self.p['b1']
            a1 = np.maximum(0, z1) # ReLU activation function
            z2 = np.dot(a1, self.p['W2']) + self.p['b2']
            exp_z = np.exp(z2)
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            # backward
            delta3 = probs
            delta3[range(len(X)), y] -= 1
            dW2 = np.dot(a1.T, delta3)
            db2 = np.sum(delta3, axis=0)
            delta2 = np.dot(delta3, self.p['W2'].T) * (a1 > 0) # derivative of ReLU
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)
            # Update parameters
            self.p['W1'] -= learning_rate * dW1
            self.p['b1'] -= learning_rate * db1
            self.p['W2'] -= learning_rate * dW2
            self.p['b2'] -= learning_rate * db2

            if epoch % 100 == 0:
                loss = self.loss(X, y)
                print("Epoch {}: loss = {}".format(epoch, loss))

def test():
    # Generate a toy dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    # Initialize a neural network
    net = TwoLayerNet(input_size=2, hidden_size=10, output_size=2)
    # Train the neural network
    net.train(X, y, num_epochs=1000)
    # Test the neural network
    probs = net.forward(X)
    predictions = np.argmax(probs, axis=1)
    print("Predictions: ", predictions)

if __name__ == '__main__':
    test()