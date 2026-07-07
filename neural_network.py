import numpy as np
from activations import relu, softmax

class NeuralNetwork:

    def __init__(self, layer_sizes):

        self.weights = []
        self.biases = []

        # Store values for backpropagation
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

        # Store input
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
        def backward(self, y_true):
        pass

        return output