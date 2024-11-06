import torch
from torch import nn

def my_layer_norm(x, gamma, beta):
    x = x - torch.mean(x, dim=-1, keepdim=True)
    # print("NOFF X =", x)

    var_x = torch.mean(x ** 2, dim=-1, keepdim=True)
    var_x = var_x + 1e-5
    var_x = torch.sqrt(var_x)
    # print("VAR X =",  var_x)

    x = x / var_x
    x = x * gamma + beta
    return x
