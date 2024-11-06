# todo: output toooo large!

import torch
from torch import nn  
# import numpy as np
from my_layer_norm import my_layer_norm

class MHABlock:
    def __init__(self, beta, gamma, Wq, Bq, Wk, Bk, Wv, Bv, pet, Wp, bias_u, bias_v, sqrt_d, Wo, Bo) -> None:
        self.beta = beta
        self.gamma = gamma
        
        self.Wq = Wq
        self.Bq = Bq
        self.Wk = Wk
        self.Bk = Bk
        self.Wv = Wv
        self.Bv = Bv
        
        self.pet = pet # positional embedding tensor
        self.Wp = Wp
        
        self.bias_u = bias_u
        self.bias_v = bias_v

        self.sqrt_d = sqrt_d

        self.Wo = Wo
        self.Bo = Bo    

    def __call__(self, x):
        # layer norm
        x = my_layer_norm(x, self.gamma, self.beta)

        # Q, K, V
        
        HEAD = 4
        D_DIV_H = x.shape[-1] // HEAD
        PET_LEN = len(self.pet)

        Q = (torch.matmul(x, self.Wq) + self.Bq).reshape((-1, HEAD, D_DIV_H))
        K = (torch.matmul(x, self.Wk) + self.Bk).reshape((-1, HEAD, D_DIV_H))
        
        V = (torch.matmul(x, self.Wv) + self.Bv).reshape((-1, HEAD, D_DIV_H))        
        # V is wrong?

        # PET slice
        pet = self.pet[(PET_LEN + 1) // 2 - len(x) : (PET_LEN - 1) // 2 + len(x)]
        P = torch.matmul(pet, self.Wp).reshape(-1, HEAD, D_DIV_H)

        print("Q =", Q)
        print("bias_u =", self.bias_u)
        print("P =", P)

        qk = torch.matmul((Q + self.bias_u).permute(1, 0, 2), K.permute(1, 2, 0))
        qp = torch.matmul((Q + self.bias_v).permute(1, 0, 2), P.permute(1, 2, 0))
        # slice qp
        qp = qp[:, :, : qk.shape[-1]]

        # print("qk.shape", qk.shape)
        # print("qp.shape", qp.shape)
        # print("V.shape", V.shape)

        print("qk =", qk)
        print("qp =", qp)
        print("V =", V)
        # print("S(qk+qp) / sqrt(d) =", ((qk + qp) / self.sqrt_d))

        # print("max qk =", qk.max())
        print("qk.shape", qk.shape)
        print("qp.shape", qp.shape)
        print("V.shape", V.shape)


        att = torch.matmul(nn.functional.softmax((qk + qp) / self.sqrt_d, dim=-1), V.permute(1, 0, 2))
        att = att.permute(1, 0, 2).reshape(-1, x.shape[-1])

        # print("v max =", V.max())
        # print("att max =", att.max())

        out = torch.matmul(att, self.Wo) + self.Bo

        return out

if __name__ == "__main__":
    import numpy as np
    beta = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.norm_self_att.mod.bias.npy"))
    gamma = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.norm_self_att.mod.weight.npy"))
    
    Wk = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5508.npy"))
    Wq = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5498.npy"))
    Wv = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5509.npy"))
    
    Bq = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.self_attn.linear_q.bias.npy"))
    Bk = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.self_attn.linear_k.bias.npy") )
    Bv = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.self_attn.linear_v.bias.npy"))
    
    pet = torch.from_numpy(np.load("param_nonquant/pet.npy").squeeze())
    Wp = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5510.npy"))
    
    bias_u = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.self_attn.pos_bias_u.npy"))
    bias_v = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.self_attn.pos_bias_v.npy"))
    sqrt_d = 6.633249759674072
    
    Wo = torch.from_numpy(np.load("param_nonquant/onnx_MatMul_5569.npy"))
    Bo = torch.from_numpy(np.load("param_nonquant/encoder.layers.0.self_attn.linear_out.bias.npy"))
    
    mha = MHABlock(beta, gamma, Wq, Bq, Wk, Bk, Wv, Bv, pet, Wp, bias_u, bias_v, sqrt_d, Wo, Bo)
    
    x = torch.rand((166, 176))
    y = mha(x)
    print(y.shape)
    print(y)
    print(y.max())