import numpy as np
from activation import sigmoid
from layernorm import layernorm

class FeedForwardBlock:
    def __init__(self, beta, gamma, W1, B1, W2, B2) -> None:
        self.beta = beta
        self.gamma = gamma
        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2


    def __call__(self, x):
        x = layernorm(x, self.beta, self.gamma)

        # linear 1
        x = np.matmul(x, self.W1) + self.B1

        # swish
        x = sigmoid(x) * x

        # linear 2
        x = np.matmul(x, self.W2) + self.B2

        return x * 0.5

if __name__ == "__main__":
    beta = np.load("param_nonquant/encoder.layers.0.norm_feed_forward1.mod.bias.npy")
    gamma = np.load("param_nonquant/encoder.layers.0.norm_feed_forward1.mod.weight.npy")

    W1 = np.load("param_nonquant/onnx_MatMul_5496.npy")
    W2 = np.load("param_nonquant/onnx_MatMul_5497.npy")
    B1 = np.load("param_nonquant/encoder.layers.0.feed_forward1.linear1.bias.npy")
    B2 = np.load("param_nonquant/encoder.layers.0.feed_forward1.linear2.bias.npy")

    # print(beta.shape)
    ff = FeedForwardBlock(beta, gamma, W1, B1, W2, B2)
    x = np.random.random((704, 176,))
    y = ff(x)
    print(y)