from optimizers import Momentum
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
optimizer = Momentum(learning_rate=0.01)

# Training Loop
batch_size = 64
epochs = 10

for epoch in range(epochs):

    epoch_loss = 0

    for i in range(0, len(x_train), batch_size):

        X_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        output = nn.forward(X_batch)

        loss = cross_entropy(y_batch, output)

        nn.backward(y_batch)

        optimizer.update(nn)

        epoch_loss += loss

    num_batches = len(x_train) // batch_size

print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / num_batches:.4f}")