import numpy as np

def batchnorm(x, beta, gamma, mean, var):
    x = np.array([
        (x[:,i] - mean) / np.sqrt(var + 1e-5) * gamma + beta for i in range(x.shape[-1])
    ]).T
    return x