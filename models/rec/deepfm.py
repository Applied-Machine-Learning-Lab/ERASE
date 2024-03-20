import torch
import torch.nn as nn
from ..layers import MLP, FactorizationMachine

class deepfm(nn.Module):

    def __init__(self, args, input_dim):
        super(deepfm, self).__init__()
        self.dims = input_dim

        self.dropout = 0.2
        self.dnn = MLP(self.dims, True, dims=[32, 16], dropout=self.dropout)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        b,f,e = x.shape
        output_fm = self.fm(x)
        x_dnn = x.reshape(b,-1)
        x_dnn = self.dnn(x_dnn)
        output = output_fm + x_dnn
        output = torch.sigmoid(output)
        return output