import numpy as np
from sigmoid import sigmoid 
from layernorm import layernorm

class MHABlock:
    def __init__(self, beta, gamma, Wq, Wk, Wv, Wo) -> None:
        self.beta = beta
        self.gamma = gamma
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        # todo: others


    def __call__(self, x):
        res = x
        # ...
        return x * 0.5 + res
