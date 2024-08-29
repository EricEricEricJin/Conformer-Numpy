import numpy as np
def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))

