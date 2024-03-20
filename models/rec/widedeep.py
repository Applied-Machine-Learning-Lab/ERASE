import torch
import torch.nn as nn

from ..layers import MLP

class widedeep(nn.Module):

    def __init__(self, args, input_dim):
        super(widedeep, self).__init__()
        self.dims = input_dim

        self.mlp = MLP(self.dims, True, dims=[32,16], dropout=0.2)
        self.linear = nn.Linear(self.dims, 1)

    def forward(self, x):
        b,f,e = x.shape
        x = x.reshape(b,-1)
        mlp_out = self.mlp(x)
        linear_out = self.linear(x)
        x = mlp_out + linear_out
        x = torch.sigmoid(x)
        return x