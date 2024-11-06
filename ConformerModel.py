import torch
from torch import nn
import os 

from ConformerBlock import ConformerBlock
from PreEncoder import PreEncoder

class ConformerModel:
    def __init__(self, prefix, mm_dict):

        import numpy as np
        def load_tensor_from_np(fp):
            return torch.from_numpy(np.load(fp))

        # Pre-Encoder
        pe_Wconv1 = load_tensor_from_np(os.path.join(prefix, "encoder.pre_encode.conv.0.weight.npy"))
        pe_Bconv1 = load_tensor_from_np(os.path.join(prefix, "encoder.pre_encode.conv.0.bias.npy"))

        pe_Wconv2 = load_tensor_from_np(os.path.join(prefix, "encoder.pre_encode.conv.2.weight.npy"))
        pe_Bconv2 = load_tensor_from_np(os.path.join(prefix, "encoder.pre_encode.conv.2.bias.npy"))
        
        pe_Wbig = load_tensor_from_np(os.path.join(prefix, "onnx_MatMul_5484.npy"))
        pe_bias = load_tensor_from_np(os.path.join(prefix, "encoder.pre_encode.out.bias.npy"))
        pe_scale = 13.266499519348145

        self.pre_encoder = PreEncoder(pe_Wconv1, pe_Bconv1, pe_Wconv2, pe_Bconv2, pe_Wbig, pe_bias, pe_scale)

        # 16 conformer blocks
        self.conformer_block_list = []
        for layer in range(16):
            self.conformer_block_list.append(ConformerBlock(prefix, layer, mm_dict[f"encoder.layers.{layer}"]))

        # Final Output
        self.decoder_conv_weight = load_tensor_from_np(os.path.join(prefix, "decoder.decoder_layers.0.weight.npy"))
        self.decoder_conv_bias = load_tensor_from_np(os.path.join(prefix, "decoder.decoder_layers.0.bias.npy"))
        print("decoder w.shape =", self.decoder_conv_weight.shape)
        print("decoder b.shape =", self.decoder_conv_bias.shape)
        # exit()

    def decoder(self, x):
        w = self.decoder_conv_weight.unsqueeze(0).permute(1, 0, 2, 3)

        x = x.unsqueeze(0).permute(0, 2, 1).unsqueeze(0)
        x = nn.functional.conv2d(x, w, bias=self.decoder_conv_bias)
        x = x.squeeze().permute(1, 0)
        x = nn.functional.log_softmax(x, dim=-1)
        return x

    def __call__(self, x):
        x = self.pre_encoder(x)
        for i in range(16):
            x = self.conformer_block_list[i](x)
        print("Before decoder x.shape =", x.shape)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    from process_param_name import get_mm_dict
    from prep_input import compute_feat

    mm_dict = get_mm_dict("param_shape.txt")
    model = ConformerModel("param_nonquant", mm_dict)

    x = compute_feat("test_wavs/0.wav")
    x = torch.from_numpy(x)

    x = model(x)
    print("final x =", x)
    print("final x.shape =", x.shape)
