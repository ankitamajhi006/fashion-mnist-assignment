import numpy as np

class Nesterov:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.learning_rate = learning_rate
        self.beta = beta

        self.vW = []
        self.vb = []

    def update(self, network):

        if len(self.vW) == 0:

            for W, b in zip(network.weights, network.biases):

                self.vW.append(np.zeros_like(W))
                self.vb.append(np.zeros_like(b))

        for i in range(len(network.weights)):

            vW_prev = self.vW[i]
            vb_prev = self.vb[i]

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * network.dW[i]
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * network.db[i]

            network.weights[i] -= self.learning_rate * (
                self.beta * vW_prev + (1 - self.beta) * network.dW[i]
            )

            network.biases[i] -= self.learning_rate * (
                self.beta * vb_prev + (1 - self.beta) * network.db[i]
            )