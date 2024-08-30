import numpy as np
from activation import sigmoid, softmax
from layernorm import layernorm

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
        # Q, K, V
        
        HEAD = 4
        D_DIV_H = x.shape[-1] // HEAD
        PET_LEN = len(self.pet)

        Q = (np.matmul(x, self.Wq) + self.Bq).reshape((-1, HEAD, D_DIV_H))
        K = (np.matmul(x, self.Wk) + self.Bk).reshape((-1, HEAD, D_DIV_H))
        V = (np.matmul(x, self.Wv) + self.Bv).reshape((-1, HEAD, D_DIV_H))        

        # PET slice
        pet = self.pet[(PET_LEN + 1) // 2 - len(x) : (PET_LEN - 1) // 2 + len(x)]
        P = np.matmul(pet, self.Wp).reshape(-1, HEAD, D_DIV_H)

        print("Q.shape", Q.shape)
        print("bias_u.shape", self.bias_u.shape)
        print("P.shape", P.shape)

        qk = np.matmul((Q + self.bias_u).transpose(1, 0, 2), K.transpose(1, 2, 0))
        qp = np.matmul((Q + self.bias_v).transpose(1, 0, 2), P.transpose(1, 2, 0))
        # slice qp
        qp = qp[:, :, : qk.shape[-1]]

        print("qk.shape", qk.shape)
        print("qp.shape", qp.shape)
        print("V.shape", V.shape)

        att = np.matmul(softmax((qk + qp) / self.sqrt_d), V.transpose(1, 0, 2))
        att = att.transpose(1, 0, 2).reshape(-1, x.shape[-1])

        out = np.matmul(att, self.Wo) + self.Bo

        return x + out


if __name__ == "__main__":
    beta = np.load("param_nonquant/encoder.layers.0.norm_self_att.mod.bias.npy")
    gamma = np.load("param_nonquant/encoder.layers.0.norm_self_att.mod.weight.npy")
    
    Wq = np.load("param_nonquant/onnx_MatMul_5508.npy")
    Wk = np.load("param_nonquant/onnx_MatMul_5509.npy")
    Wv = np.load("param_nonquant/onnx_MatMul_5510.npy")
    
    Bq = np.load("param_nonquant/encoder.layers.0.self_attn.linear_q.bias.npy")
    Bk = np.load("param_nonquant/encoder.layers.0.self_attn.linear_k.bias.npy") 
    Bv = np.load("param_nonquant/encoder.layers.0.self_attn.linear_v.bias.npy")
    
    pet = np.load("param_nonquant/pet.npy").squeeze()
    Wp = np.load("param_nonquant/onnx_MatMul_5498.npy")
    
    bias_u = np.load("param_nonquant/encoder.layers.0.self_attn.pos_bias_u.npy")
    bias_v = np.load("param_nonquant/encoder.layers.0.self_attn.pos_bias_v.npy")
    sqrt_d = 6.633249759674072
    
    Wo = np.load("param_nonquant/onnx_MatMul_5569.npy")
    Bo = np.load("param_nonquant/encoder.layers.0.self_attn.linear_out.bias.npy")
    
    mha = MHABlock(beta, gamma, Wq, Bq, Wk, Bk, Wv, Bv, pet, Wp, bias_u, bias_v, sqrt_d, Wo, Bo)
    
    x = np.random.random((166, 176))
    y = mha(x)
    print(y.shape)