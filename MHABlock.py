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
        D_DIV_H = x.shape[-1] / HEAD
        PET_LEN = len(self.pet)

        Q = (np.matmul(x, self.Wq) + self.Bq).reshape((-1, HEAD, D_DIV_H))
        K = (np.matmul(x, self.Wk) + self.Bk).reshape((-1, HEAD, D_DIV_H))
        V = (np.matmul(x, self.Wv) + self.Bv).reshape((-1, HEAD, D_DIV_H))        

        # PET slice
        P = np.matmul(self.pet[(PET_LEN + 1) // 2 - len(x) : (PET_LEN - 1) // 2 + len(x)], self.Wp).reshape(-1, HEAD, D_DIV_H).transpose(1, 2, 0)

        qk = np.matmul((Q + self.bias_u).transpose(1, 0, 2), K.transpose(1, 2, 0))
        qp = np.matmul((Q + self.bias_v).transpose(1, 0, 2), P.transpose(1, 2, 0))

        att = np.matmul(V.transpose(1, 0, 2), softmax((qk + qp) / self.sqrt_d))
        att = att.transpose(1, 0, 2).reshape(-1, x.shape[-1])

        out = np.matmul(att, self.Wo) + self.Bo

        return x + out
