import torch
import torch.nn as nn
from ..layers import MLP, FactorizationMachine

class fm(nn.Module):

    def __init__(self, args, input_dim):
        super(fm, self).__init__()
        self.dims = input_dim

        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        b,f,e = x.shape
        output_fm = self.fm(x)
        x = torch.sigmoid(output_fm)
        return x