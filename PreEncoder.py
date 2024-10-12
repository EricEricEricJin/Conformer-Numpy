import torch
from torch import nn

class PreEncoder:
    def __init__(self, Wconv1, Bconv1, Wconv2, Bconv2, Wbig, bias, scale) -> None:
        self.Wconv1 = Wconv1
        self.Bconv1 = Bconv1
        self.Wconv2 = Wconv2
        self.Bconv2 = Bconv2

        self.Wbig = Wbig
        self.bias = bias
        self.scale = scale 

        # conv 1

    def __call__(self, x):
        # Wconv1.shape = torch.Size([176, 1, 3, 3])
        # Bconv1.shape = torch.Size([176])
        # Wconv2.shape = torch.Size([176, 176, 3, 3])
        # Bconv2.shape = torch.Size([176])
        # Wbig.shape = torch.Size([3520, 176])
        # bias.shape = torch.Size([176])
        # x.shape = torch.Size([1, 80, 663]) --> use squeezed x: 663, 80

        # x.shape = (L, 80,)
        # assert(len(x.shape) == 2 and x.shape[1] = 80)
        
        # conv 1
        # pading: 1,1,1,1
        # stride: 2,2

        # W: 176, 1, 3, 3
        x = x.unsqueeze(0).unsqueeze(0) # 1, 1, 663, 80
        x = nn.functional.conv2d(x, self.Wconv1, bias=self.Bconv1, stride=[2, 2], padding=[1, 1, 1, 1])

        # relu
        x = nn.functional.relu(x)

        # conv 2
        # W: 176, 176, 3, 3
        # x: 1, 176, 332, 40
        x = nn.functional.conv2d(x, self.Wconv2, bias=self.Bconv2, stride=[2, 2], padding=[1, 1, 1, 1])

        # relu and reshape
        x = nn.functional.relu(x)
        x = x.reshape() # todo


if __name__ == "__main__":
    import numpy as np
    def load_tensor_from_np(fp):
        return torch.from_numpy(np.load(fp))
    
    Wconv1 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.0.weight.npy")
    Bconv1 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.0.bias.npy")

    Wconv2 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.2.weight.npy")
    Bconv2 = load_tensor_from_np("param_nonquant/encoder.pre_encode.conv.2.bias.npy")
    
    Wbig = load_tensor_from_np("param_nonquant/onnx_MatMul_5484.npy")
    bias = load_tensor_from_np("param_nonquant/encoder.pre_encode.out.bias.npy")
    scale = 13.266499519348145

    x = load_tensor_from_np("feat_0.wav.npy")
    x = x.squeeze().permute(1, 0)

    print("Wconv1.shape =", Wconv1.shape)
    print("Bconv1.shape =", Bconv1.shape)
    print("Wconv2.shape =", Wconv2.shape)
    print("Bconv2.shape =", Bconv2.shape)

    print("Wbig.shape =", Wbig.shape)
    print("bias.shape =", bias.shape)
    print("x.shape =", x.shape)