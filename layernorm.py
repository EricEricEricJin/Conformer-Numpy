import numpy as np

def layernorm(x, beta, gamma):
    x = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))
    x = x * gamma + beta
    return x