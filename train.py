from keras.datasets import fashion_mnist
import numpy as np
from neural_network import NeuralNetwork

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(60000, 784) / 255.0

nn = NeuralNetwork(
    input_size=784,
    hidden_size=128,
    output_size=10
)

output = nn.forward(X_train[:5])

print("Output shape:", output.shape)
print(output)