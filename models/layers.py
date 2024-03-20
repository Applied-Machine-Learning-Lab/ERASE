import torch.nn as nn
import torch
from itertools import combinations

class MLP(nn.Module):

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class CrossNetwork(nn.Module):
    """CrossNetwork  mentioned in the DCN paper.

    Args:
        input_dim (int): input dim of input tensor
    
    Shape:
        - Input: `(batch_size, *)`
        - Output: `(batch_size, *)`
        
    """

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x
    
class SENETLayer(nn.Module):
    """
    A weighted feature gating system in the SENet paper
    Args:
        num_fields (int): number of feature fields

    Shape:
        - num_fields: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields/ reduction_ratio))
        self.mlp = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(reduced_size, num_fields, bias=False),
                                 nn.ReLU())
    def forward(self, x):
        z = torch.mean(x, dim=-1, out=None)
        a = self.mlp(z)
        v = x*a.unsqueeze(-1)
        return v
    
class BiLinearInteractionLayer(nn.Module):
    """
    Bilinear feature interaction module, which is an improved model of the FFM model
     Args:
        num_fields (int): number of feature fields
        bilinear_type(str): the type bilinear interaction function
    Shape:
        - num_fields: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """
    def __init__(self, input_dim, num_fields, bilinear_type = "field_interaction"):
        super(BiLinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(input_dim, input_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i,j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, x):
        feature_emb = torch.split(x, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i)*v_j for v_i, v_j in combinations(feature_emb, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb[i])*feature_emb[j] for i,j in combinations(range(len(feature_emb)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0])*v[1] for i,v in enumerate(combinations(feature_emb, 2))]
        return torch.cat(bilinear_list, dim=1)

class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1) if reduce_sum
                  tensor of size (batch_size, embed_dim) else   
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix