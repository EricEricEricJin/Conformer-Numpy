import numpy as np
import scipy
import scipy.signal

from layernorm import layernorm
from batchnorm import batchnorm
from activation import sigmoid

class ConvolutionBlock:
    def __init__(self, beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_beta, bn_gamma, bn_mean, bn_var, Wpt2, Bpt2) -> None:
        self.beta = beta
        self.gamma = gamma

        # conv pt 1
        self.Wpt1 = Wpt1
        self.Bpt1 = Bpt1

        # conv dp
        self.Wdp = Wdp
        self.Bdp = Bdp

        # batch norm
        self.bn_beta = bn_beta
        self.bn_gamma = bn_gamma
        self.bn_mean = bn_mean
        self.bn_var = bn_var

        # conv pt 2
        self.Wpt2 = Wpt2
        self.Bpt2 = Bpt2

    @staticmethod
    def conv_pt(x, W, B):
        # x: 176, m
        # W: 352, 176, 1
        # B: 352,
        # out: 352, m

        assert(len(x.shape) == 2)
        assert(len(W.shape) == 3)
        assert(len(B.shape) == 1)
        print(x.shape, W.shape)
        assert(x.shape[0] == W.shape[1])
        assert(W.shape[0] == B.shape[0])

        y = np.array([scipy.signal.correlate2d(x, W[i], mode="valid") + B[i] for i in range(len(W))])
        y = np.squeeze(y, axis=1)
        assert(y.shape == (B.shape[0], x.shape[1]))
        return y

    @staticmethod
    def conv_dp(x, W, B):
        # x: 176, m
        # W: 176, 1, k
        # B: 176,
        # out: 176, m-k+1

        y = np.array([scipy.signal.correlate(x[i], W[i][0], mode="valid") + B[i] for i in range(len(W))])
        assert(y.shape == (x.shape[0], x.shape[1] - W.shape[2] + 1))
        return y

    def __call__(self, x):
        x = layernorm(x, self.beta, self.gamma)
        
        # pointwise conv
        x = x.transpose()
        x = self.conv_pt(x, self.Wpt1, self.Bpt1)

        # split
        x0, x1 = np.split(x, 2, axis=0)
        x = x0 * sigmoid(x1)

        # padding
        # in: (176, 166)
        # out: (176, 196)
        print("x.shape", x.shape)
        x = np.pad(x, ((0, 0), (15, 15)))

        # dp conv
        x = self.conv_dp(x, self.Wdp, self.Bdp)

        # batch norm
        x = batchnorm(x, self.bn_beta, self.bn_gamma, self.bn_mean, self.bn_var)

        x = sigmoid(x) * x
        x = self.conv_pt(x, self.Wpt2, self.Bpt2)
        
        x = x.transpose()
                
        return x


if __name__ == "__main__":
    beta = np.load("param_nonquant/encoder.layers.0.norm_conv.mod.weight.npy")
    gamma = np.load("param_nonquant/encoder.layers.0.norm_conv.mod.bias.npy")

    Wpt1 = np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv1.weight.npy")
    Bpt1 = np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv1.bias.npy")
    
    Wdp = np.load("param_nonquant/encoder.layers.0.conv.depthwise_conv.weight.npy")
    Bdp = np.load("param_nonquant/encoder.layers.0.conv.depthwise_conv.bias.npy")
    
    bn_gamma = np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.weight.npy")
    bn_beta = np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.bias.npy")
    bn_mean = np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.running_mean.npy")
    bn_var = np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.running_var.npy")

    Wpt2 = np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv2.weight.npy")
    Bpt2 = np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv2.bias.npy")
 
    for x in (beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_gamma, bn_beta, bn_mean, bn_var, Wpt2, Bpt2):
        print(x.shape)

    # print(beta.shape)
    conv = ConvolutionBlock(beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_beta, bn_gamma, bn_mean, bn_var, Wpt2, Bpt2)
    x = np.random.random((166, 176))
    y = conv(x)
    print(y.shape)
    print(y)