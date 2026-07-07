import numpy as np

def one_hot(y, num_classes=10):
    one_hot_labels = np.zeros((len(y), num_classes))
    one_hot_labels[np.arange(len(y)), y] = 1
    return one_hot_labels