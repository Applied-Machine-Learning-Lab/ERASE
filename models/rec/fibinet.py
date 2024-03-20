import torch
import torch.nn as nn
from ..layers import MLP, SENETLayer, BiLinearInteractionLayer

class fibinet(nn.Module):

    def __init__(self, args, input_dim):
        super(fibinet, self).__init__()

        # if you change the embedding_size, this value should be changed too
        embedding_dim = args.embedding_dim
        self.dims = input_dim
        self.num_fields = self.dims // embedding_dim
        self.senet_layer = SENETLayer(self.num_fields, reduction_ratio=3)
        self.bilinear_interaction = BiLinearInteractionLayer(embedding_dim, self.num_fields, bilinear_type="field_interaction")
        self.hidden_size = self.num_fields * (self.num_fields - 1) * embedding_dim
        self.mlp = MLP(self.hidden_size, True, dims=[32,16], dropout=0.2)

    def forward(self, x):
        b,f,e = x.shape
        embed_senet = self.senet_layer(x)
        embed_bi1 = self.bilinear_interaction(x)
        embed_bi2 = self.bilinear_interaction(embed_senet)
        shallow_part = torch.flatten(torch.cat([embed_bi1, embed_bi2], dim=1), start_dim=1)
        mlp_out = self.mlp(shallow_part)
        output = torch.sigmoid(mlp_out)
        return output
