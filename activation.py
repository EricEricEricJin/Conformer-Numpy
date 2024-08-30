import numpy as np
def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax(x):
    return np.exp(x) / sum(np.exp(x))