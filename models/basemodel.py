from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.module import Module
from utils.utils import get_model

class BaseModel(nn.Module):
    def __init__(self, args, backbone_model_name, fs, es, unique_values, features):
        super(BaseModel, self).__init__()
        # embedding table
        self.embedding = nn.Embedding(sum(unique_values), embedding_dim = args.embedding_dim)
        torch.nn.init.normal_(self.embedding.weight.data, mean=0, std=0.01)
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))

        self.input_dims = args.embedding_dim * len(unique_values)

        self.bb = get_model(backbone_model_name, 'rec')(args, self.input_dims) # backbone model name
        self.fs = get_model(fs, 'fs')(args, unique_values, features) # feature selection method
        self.es = get_model(es, 'es')() # embedding search method
        self.args = args

    def forward(self, x, current_epoch, current_step):
        raw_x = x.clone().detach()
        x = self.embedding(x + x.new_tensor(self.offsets))
        x = self.es(x)
        x = self.fs(x, current_epoch, current_step, raw_data = raw_x)
        x = self.bb(x)
        return x
    
    def set_optimizer(self):
        optimizer_bb = torch.optim.Adam([params for name,params in self.named_parameters() if ('fs' not in name and 'es' not in name) or 'bb' in name], lr = self.args.learning_rate)
        
        if [params for name,params in self.named_parameters() if 'fs' in name] != []:
            optimizer_fs = torch.optim.Adam([params for name,params in self.named_parameters() if 'fs' in name and 'bb' not in name], lr = self.args.learning_rate)
        else:
            optimizer_fs = None
        
        if [params for name,params in self.named_parameters() if 'es' in name] != []:
            optimizer_es = torch.optim.Adam([params for name,params in self.named_parameters() if 'es' in name and 'bb' not in name], lr = self.args.learning_rate)
        else:
            optimizer_es = None
        return {'optimizer_bb': optimizer_bb, 'optimizer_fs': optimizer_fs, 'optimizer_es': optimizer_es}
    