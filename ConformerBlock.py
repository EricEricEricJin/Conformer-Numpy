from FeedForwardBlock import FeedForwardBlock
from MHABlock import MHABlock
from ConvolutionBlock import ConvolutionBlock
import numpy as np
import os

os.path.join

class ConformerBlock:
    def __init__(self, prefix, layer, mm_dict) -> None:
        # FF1
        beta = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward1.mod.bias.npy"))
        gamma = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward1.mod.weight.npy"))
        W1 = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward1.linear1.weight']}.npy"))
        W2 = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward1.linear2.weight']}.npy"))
        B1 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward1.linear1.bias.npy"))
        B2 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward1.linear2.bias.npy"))
        self.FF1 = FeedForwardBlock(beta, gamma, W1, B1, W2, B2)
        
        # MHA
        # todo
        
        # CONV
        beta = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_conv.mod.weight.npy"))
        gamma = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_conv.mod.bias.npy"))
        Wpt1 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv1.weight.npy"))
        Bpt1 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv1.bias.npy"))
        Wdp = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.depthwise_conv.weight.npy"))
        Bdp = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.depthwise_conv.bias.npy"))
        bn_gamma = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.weight.npy"))
        bn_beta = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.bias.npy"))
        bn_mean = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.running_mean.npy"))
        bn_var = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.running_var.npy"))
        Wpt2 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv2.weight.npy"))
        Bpt2 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv2.bias.npy"))
        self.CONV = ConvolutionBlock(beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_beta, bn_gamma, bn_mean, bn_var, Wpt2, Bpt2)

        # FF2
        

    def __call__(x):
        pass