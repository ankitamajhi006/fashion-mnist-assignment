import numpy as np

class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, network):

        for i in range(len(network.weights)):
            network.weights[i] -= self.learning_rate * network.dW[i]
            network.biases[i] -= self.learning_rate * network.db[i]


class Momentum:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.learning_rate = learning_rate
        self.beta = beta

        self.vW = []
        self.vb = []

    def update(self, network):

        # Initialize velocity vectors
        if len(self.vW) == 0:

            for W, b in zip(network.weights, network.biases):

                self.vW.append(np.zeros_like(W))
                self.vb.append(np.zeros_like(b))

        # Update weights
        for i in range(len(network.weights)):

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * network.dW[i]
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * network.db[i]

            network.weights[i] -= self.learning_rate * self.vW[i]
            network.biases[i] -= self.learning_rate * self.vb[i]