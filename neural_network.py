import numpy as np
from activations import relu, softmax

class NeuralNetwork:

    def __init__(self, layer_sizes):

        self.weights = []
        self.biases = []

        self.activations = []
        self.z_values = []

        self.dW = []
        self.db = []

        for i in range(len(layer_sizes) - 1):

            W = np.random.randn(
                layer_sizes[i],
                layer_sizes[i + 1]
            ) * np.sqrt(1.0 / layer_sizes[i])

            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):

        self.activations = [X]
        self.z_values = []

        A = X

        # Hidden layers
        for i in range(len(self.weights) - 1):

            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_values.append(Z)

            A = relu(Z)
            self.activations.append(A)

        # Output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.z_values.append(Z)

        output = softmax(Z)
        self.activations.append(output)

        return output

    def backward(self, y_true):

        m = y_true.shape[0]

        self.dW = []
        self.db = []

        # Output layer error
        delta = self.activations[-1] - y_true

        # Output layer gradients
        dW = np.dot(self.activations[-2].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m

        self.dW.insert(0, dW)
        self.db.insert(0, db)

        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):

            delta = np.dot(delta, self.weights[i + 1].T) * (self.z_values[i] > 0)

            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.dW.insert(0, dW)
            self.db.insert(0, db)