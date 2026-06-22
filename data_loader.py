from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

plt.figure(figsize=(10,5))

for i in range(10):
    index = list(y_train).index(i)

    plt.subplot(2,5,i+1)
    plt.imshow(X_train[index], cmap="gray")
    plt.title(class_names[i])
    plt.axis("off")

plt.tight_layout()
plt.show()