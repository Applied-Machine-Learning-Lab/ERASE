import torch
import torch.nn as nn
from ..layers import MLP, CrossNetwork

class dcn(nn.Module):

    def __init__(self, args, input_dim):
        super(dcn, self).__init__()
        self.dims = input_dim

        self.cn = CrossNetwork(self.dims, num_layers=2)
        self.mlp = MLP(self.dims, False, dims=[32,16], dropout=0.2)
        self.linear = nn.Linear(self.dims + 16, 1)

    def forward(self, x):
        b,f,e = x.shape
        x = x.reshape(b,-1)
        cn_out = self.cn(x)
        mlp_out = self.mlp(x)
        x = torch.cat([cn_out, mlp_out], dim=1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x