from optimizers import SGD
from loss import cross_entropy
from data_loader import load_data
from neural_network import NeuralNetwork
from utils import one_hot

# Load dataset
(x_train, y_train), (x_test, y_test) = load_data()

# Flatten and normalize
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

# One-hot encode labels
y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Create neural network
nn = NeuralNetwork([784, 128, 64, 32, 10])
optimizer = SGD(learning_rate=0.01)

# Training Loop
epochs = 10

for epoch in range(epochs):

    # Forward pass
    output = nn.forward(x_train[:5])

    # Compute loss
    loss = cross_entropy(y_train[:5], output)

    # Backward pass
    nn.backward(y_train[:5])

    # Update weights
    optimizer.update(nn)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")