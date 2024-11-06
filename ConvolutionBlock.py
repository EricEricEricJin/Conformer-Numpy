import torch 
from torch import nn
from my_layer_norm import my_layer_norm

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
        y_shape = (B.shape[0], x.shape[1])

        assert(len(x.shape) == 2)
        assert(len(W.shape) == 3)
        assert(len(B.shape) == 1)
        print(x.shape, W.shape)
        assert(x.shape[0] == W.shape[1])
        assert(W.shape[0] == B.shape[0])

        # y = np.array([scipy.signal.correlate2d(x, W[i], mode="valid") + B[i] for i in range(len(W))])
        # y = np.squeeze(y, axis=1)

        x = x.unsqueeze(0).unsqueeze(0)  # (176, m) -> (1, 1, 176, m)
        W = W.unsqueeze(1)  # (352, 176, 1) -> (352, 1, 176, 1)
        print("pt conv", "x shape =", x.shape, "W shape =", W.shape, "B shape =", B.shape)
        y = nn.functional.conv2d(x, W, bias=B, stride=1).squeeze()
        print("y.shape =", y.shape)
        assert(y.shape == y_shape)
        return y

    @staticmethod
    def conv_dp(x, W, B):
        # x: 176, m
        # W: 176, 1, k
        # B: 176,
        # out: 176, m-k+1
        y_shape = (x.shape[0], x.shape[1] - W.shape[2] + 1)

        # y = np.array([scipy.signal.correlate(x[i], W[i][0], mode="valid") + B[i] for i in range(len(W))])
        x = x.unsqueeze(0).permute(1, 0, 2).unsqueeze(0)    # x: (1, 176, m) -> (176, 1, m) -> (1, 176, 1, m)
        W = W.unsqueeze(0).permute(1, 0, 2, 3)              # W: (176, 1, k) -> (1, 176, 1, k) -> (176, 1, 1, k)
        
        # input - input tensor of shape (minibatch,in_channels,iH,iW)(minibatch,in_channels,iH,iW)
        # weight - filters of shape (out_channels,in_channelsgroups,kH,kW)(out_channels,groupsin_channels​,kH,kW)
        print("x.shape =", x.shape, "W.shape =", W.shape)
        y = nn.functional.conv2d(x, W, bias=B, stride=1, groups=x.shape[1])
        y = y.squeeze()
        
        print("y shape =", y.shape)
        assert(y.shape == y_shape)
        return y

    def __call__(self, x):
        # x = nn.functional.layer_norm(x, self.beta.shape, weight=self.gamma, bias=self.beta)
        x = my_layer_norm(x, self.gamma, self.beta)
        
        # pointwise conv
        x = x.permute(1, 0)
        x = self.conv_pt(x, self.Wpt1, self.Bpt1)

        print("after 1st pt conv", x.max())

        # split
        x0, x1 = torch.split(x, x.shape[0] // 2, dim=0)
        x = x0 * nn.functional.sigmoid(x1)

        # padding
        # in: (176, 166)
        # out: (176, 196)
        # print("x.shape", x.shape)
        x = nn.functional.pad(x, (15, 15, 0, 0))

        # dp conv
        # print("before dp conv", x.max())

        x = self.conv_dp(x, self.Wdp, self.Bdp)

        # print("after dp conv", x.max())
        # print(f"BN mean = {self.bn_mean}, var = {self.bn_var}, weight = {self.bn_gamma}, bias = {self.bn_gamma}")
        # batch norm
        # x = batchnorm(x, self.bn_beta, self.bn_gamma, self.bn_mean, self.bn_var)
        x = nn.functional.batch_norm(x.permute(1, 0), running_mean=self.bn_mean, running_var=self.bn_var, 
                                     weight=self.bn_gamma, bias=self.bn_beta).permute(1, 0)
        print("after bn", x.max())
        x = nn.functional.sigmoid(x) * x
        x = self.conv_pt(x, self.Wpt2, self.Bpt2)

        x = x.permute(1, 0)
                
        return x


if __name__ == "__main__":
    import numpy as np
    beta = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.norm_conv.mod.weight.npy"))
    gamma = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.norm_conv.mod.bias.npy"))

    Wpt1 = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv1.weight.npy"))
    Bpt1 = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv1.bias.npy"))
    
    Wdp = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.depthwise_conv.weight.npy"))
    Bdp = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.depthwise_conv.bias.npy"))
    
    bn_gamma = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.weight.npy"))
    bn_beta = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.bias.npy"))
    bn_mean = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.running_mean.npy"))
    bn_var = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.batch_norm.mod.running_var.npy"))

    Wpt2 = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv2.weight.npy"))
    Bpt2 = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.conv.pointwise_conv2.bias.npy"))
 
    for x in (beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_gamma, bn_beta, bn_mean, bn_var, Wpt2, Bpt2):
        print(x.shape)

    # print(beta.shape)
    conv = ConvolutionBlock(beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_beta, bn_gamma, bn_mean, bn_var, Wpt2, Bpt2)
    x = torch.rand((166, 176))
    y = conv(x)
    print(y.shape)
    print(y)