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
        beta = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_self_att.mod.bias.npy"))
        gamma = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_self_att.mod.weight.npy"))
        
        Wq = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_q.weight']}.npy"))
        Wk = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_k.weight']}.npy"))
        Wv = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_v.weight']}.npy"))
        
        Bq = np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_q.bias.npy"))
        Bk = np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_k.bias.npy") )
        Bv = np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_v.bias.npy"))
        
        pet = np.load(os.path.join(prefix, "pet.npy")).squeeze()
        Wp = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_pos.weight']}.npy"))
        
        bias_u = np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.pos_bias_u.npy"))
        bias_v = np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.pos_bias_v.npy"))
        
        # Write in file later
        ##################
        sqrt_d = 6.633249759674072
        ##################
        
        Wo = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_out.weight']}.npy"))
        Bo = np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_out.bias.npy"))

        self.MHA = MHABlock(beta, gamma, Wq, Bq, Wk, Bk, Wv, Bv, pet, Wp, bias_u, bias_v, sqrt_d, Wo, Bo)
        
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
        beta = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward2.mod.bias.npy"))
        gamma = np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward2.mod.weight.npy"))
        W1 = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward2.linear1.weight']}.npy"))
        W2 = np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward2.linear2.weight']}.npy"))
        B1 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward2.linear1.bias.npy"))
        B2 = np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward2.linear2.bias.npy"))
        self.FF2 = FeedForwardBlock(beta, gamma, W1, B1, W2, B2)

    def __call__(self, x):
        x = x + self.FF1(x)
        x = x + self.MHA(x)
        x = x + self.CONV(x)
        x = x + self.FF2(x)
        return x

if __name__ == "__main__":
    
    from process_param_name import get_mm_dict

    prefix = "param_nonquant"
    layer = 0
    mm_dict = get_mm_dict("param_shape.txt")
    print(mm_dict)
    cb = ConformerBlock(prefix, layer, mm_dict["encoder.layers.0"])
    x = np.random.random((166, 176))
    print(cb(x).shape)