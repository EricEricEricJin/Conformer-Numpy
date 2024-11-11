from FeedForwardBlock import FeedForwardBlock
from MHABlock import MHABlock
from ConvolutionBlock import ConvolutionBlock
import numpy as np
import os
import torch
from my_layer_norm import my_layer_norm

os.path.join

class ConformerBlock:
    def __init__(self, prefix, layer, mm_dict) -> None:
        # FF1
        beta = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward1.mod.bias.npy")))
        gamma = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward1.mod.weight.npy")))
        W1 = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward1.linear1.weight']}.npy")))
        W2 = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward1.linear2.weight']}.npy")))
        B1 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward1.linear1.bias.npy")))
        B2 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward1.linear2.bias.npy")))
        self.FF1 = FeedForwardBlock(beta, gamma, W1, B1, W2, B2)
        
        # MHA
        beta = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_self_att.mod.bias.npy")))
        gamma = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_self_att.mod.weight.npy")))
        
        Wq = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_q.weight']}.npy")))
        Wk = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_k.weight']}.npy")))
        Wv = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_v.weight']}.npy")))
        
        Bq = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_q.bias.npy")))
        Bk = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_k.bias.npy") ))
        Bv = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_v.bias.npy")))
        
        pet = torch.from_numpy(np.load(os.path.join(prefix, "pet.npy")).squeeze())
        Wp = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_pos.weight']}.npy")))
        
        bias_u = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.pos_bias_u.npy")))
        bias_v = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.pos_bias_v.npy")))
        
        # Write in file later
        ##################
        sqrt_d = 6.633249759674072
        ##################
        
        Wo = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['self_attn.linear_out.weight']}.npy")))
        Bo = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.self_attn.linear_out.bias.npy")))

        self.MHA = MHABlock(beta, gamma, Wq, Bq, Wk, Bk, Wv, Bv, pet, Wp, bias_u, bias_v, sqrt_d, Wo, Bo)
        
        # CONV
        beta = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_conv.mod.bias.npy")))
        gamma = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_conv.mod.weight.npy")))
        Wpt1 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv1.weight.npy")))
        Bpt1 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv1.bias.npy")))
        Wdp = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.depthwise_conv.weight.npy")))
        Bdp = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.depthwise_conv.bias.npy")))
        bn_gamma = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.weight.npy")))
        bn_beta = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.bias.npy")))
        bn_mean = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.running_mean.npy")))
        bn_var = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.batch_norm.mod.running_var.npy")))
        Wpt2 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv2.weight.npy")))
        Bpt2 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.conv.pointwise_conv2.bias.npy")))
        self.CONV = ConvolutionBlock(beta, gamma, Wpt1, Bpt1, Wdp, Bdp, bn_beta, bn_gamma, bn_mean, bn_var, Wpt2, Bpt2)

        # FF2
        beta = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward2.mod.bias.npy")))
        gamma = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_feed_forward2.mod.weight.npy")))
        W1 = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward2.linear1.weight']}.npy")))
        W2 = torch.from_numpy(np.load(os.path.join(prefix, f"onnx_MatMul_{mm_dict['feed_forward2.linear2.weight']}.npy")))
        B1 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward2.linear1.bias.npy")))
        B2 = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.feed_forward2.linear2.bias.npy")))
        self.FF2 = FeedForwardBlock(beta, gamma, W1, B1, W2, B2)

        # Norm Out
        self.norm_out_gamma = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_out.mod.weight.npy")))
        self.norm_out_beta = torch.from_numpy(np.load(os.path.join(prefix, f"encoder.layers.{layer}.norm_out.mod.bias.npy")))

    def __call__(self, x):
        print("INPUT", x)
        x = x + self.FF1(x)
        print("FF1", x)
        
        x = x + self.MHA(x)
        print("MHA", x)
        
        x = x + self.CONV(x)
        print("CONV", x)
        
        x = x + self.FF2(x)
        print("FF2", x)
        
        x = my_layer_norm(x, self.norm_out_gamma, self.norm_out_beta)

        return x

if __name__ == "__main__":
    
    from process_param_name import get_mm_dict

    prefix = "param_nonquant"
    layer = 0
    mm_dict = get_mm_dict("param_shape.txt")
    print(mm_dict)
    cb = ConformerBlock(prefix, layer, mm_dict["encoder.layers.0"])
    # x =  torch.from_numpy(np.random.random((166, 176), dtype=np.float32))


    from PreEncoder import PreEncoder
    import prep_input
    def load_tensor_from_np(fp):
        return torch.from_numpy(np.load(fp))
    
    Wconv1 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.0.weight.npy")
    Bconv1 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.0.bias.npy")

    Wconv2 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.2.weight.npy")
    Bconv2 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.2.bias.npy")
    
    Wbig = load_tensor_from_np("param_nonquant/onnx_MatMul_5484.npy")
    bias = load_tensor_from_np("param_nonquant/encoder.pre_encode.out.bias.npy")
    scale = 13.266499519348145

    print("Wconv1.shape =", Wconv1.shape)
    print("Bconv1.shape =", Bconv1.shape)
    print("Wconv2.shape =", Wconv2.shape)
    print("Bconv2.shape =", Bconv2.shape)

    print("Wbig.shape =", Wbig.shape)
    print("bias.shape =", bias.shape)

    x = prep_input.compute_feat("test_wavs/0.wav")
    x = torch.from_numpy(x)
    # print("feat.shape", x.shape)

    pe = PreEncoder(Wconv1, Bconv1, Wconv2, Bconv2, Wbig, bias, scale)
    x = pe(x)

    x = cb(x)

    print("final x =", x)
    print(x.shape)