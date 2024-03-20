import torch.nn as nn
import torch.nn.functional as F
import torch

# class mlp(nn.Module):
#     def __init__(self, input_size, hidden_size = 16, output_size = 16, dropout = 0.2):
#         super(mlp, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.dropout1 = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.dropout2 = nn.Dropout(dropout)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#         self.output_layer = nn.Linear(output_size, 1)
#         self.init_weights()

#         # optimizer
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, mean=0, std=0.01)
#                 nn.init.constant_(m.bias.data, 0)

#     def forward(self, x):
#         b = x.shape[0]
#         x = x.reshape(b, -1)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         x = self.output_layer(x)
#         x = torch.sigmoid(x)
#         return x
    
class mlp(nn.Module):
    def __init__(self, args, input_dim, embed_dims = [16,16], dropout = 0.2, output_layer=True):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        b = x.shape[0]
        x = x.reshape(b,-1)
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        x = torch.sigmoid(x)
        return x