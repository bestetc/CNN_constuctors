''' Module contains custom activation function'''

import torch
from torch import nn
import torch.nn.functional as F

def mish(x):
    ''' Applies the mish function element-wise 
    
    See more in Mish class docstring
    '''
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    ''' Applies the mish function element-wise
    
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    where x - tensor of any dimension
        
    See Also
    --------
    https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)
    