import torch 
from torch import nn
from my_layer_norm import my_layer_norm

class FeedForwardBlock:
    def __init__(self, beta, gamma, W1, B1, W2, B2) -> None:
        self.beta = beta
        self.gamma = gamma
        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2

    def __call__(self, x):
        print(f"x.shape = {x.shape}, beta.shape = {self.beta.shape}, gamma.shape = {self.gamma.shape}")

        # x = nn.functional.layer_norm(x, self.beta.shape, weight=self.gamma, bias=self.beta)
        print("before norm", x.min(), x.max())
        x = my_layer_norm(x, self.gamma, self.beta)
        print("norm_x", x.min(), x.max())
        
        # linear 1
        x = torch.matmul(x, self.W1) + self.B1
        print("after mm1 x =", x.max())

        # swish
        x = nn.functional.silu(x)
        print("after silu x =", x.max())

        # linear 2
        x = torch.matmul(x, self.W2) + self.B2
        print("after mm2 x =", x.max())

        x = x * 0.5
        return x

if __name__ == "__main__":
    import numpy as np
    beta = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.norm_feed_forward1.mod.bias.npy"))
    gamma = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.norm_feed_forward1.mod.weight.npy"))

    print(f"beta = {beta}")
    print(f"gamma = {gamma}")

    W1 = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5496.npy"))
    W2 = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5497.npy"))
    B1 = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.feed_forward1.linear1.bias.npy"))
    B2 = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.feed_forward1.linear2.bias.npy"))
    print("W1 =", W1)

    # print(beta.shape)
    ff = FeedForwardBlock(beta, gamma, W1, B1, W2, B2)
    x = torch.rand((704, 176,)) * 1000.0
    y = ff(x)
    print(y)
