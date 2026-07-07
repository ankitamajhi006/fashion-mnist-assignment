class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, network):

        for i in range(len(network.weights)):
            network.weights[i] -= self.learning_rate * network.dW[i]
            network.biases[i] -= self.learning_rate * network.db[i]