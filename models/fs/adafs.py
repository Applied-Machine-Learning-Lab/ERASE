import torch
import torch.nn as nn
from models.layers import MLP

class adafs(nn.Module):

    def __init__(self, args, unique_values, features):
        super(adafs, self).__init__()

        self.pretrain_epoch = args.fs_config[args.fs]['pretrain_epoch']
        self.feature_num = len(unique_values)
        self.batchnorm_bb = nn.BatchNorm1d(args.embedding_dim) # add the _bb to add this parameter in bb optimizer
        self.hidden_size = args.fs_config[args.fs]['hidden_size']
        self.dropout = args.fs_config[args.fs]['dropout']
        self.mlp = MLP(self.feature_num * args.embedding_dim, False, [self.hidden_size, self.feature_num], self.dropout)
        self.mlp_bb = MLP(self.feature_num * args.embedding_dim, False, [self.hidden_size, self.feature_num], self.dropout)
        self.mode = 'train'
        if self.mode == 'retrain':
            raise Exception('adafs should not be used in retrain mode')
        self.optimizer_method = 'normal'
        self.update_frequency = args.fs_config[args.fs]['update_frequency']

        self.load_checkpoint = False
    
    def forward(self, x, current_epoch, current_step, raw_data):
        b,f,e = x.shape
        if self.optimizer_method == 'darts':
            if current_epoch is not None and current_epoch < self.pretrain_epoch: # current_epoch not None (in training or validation) and current_epoch <= self.pretrain_epoch
                x = x.transpose(1,2)
                x = self.batchnorm_bb(x)
                return x.transpose(1,2)
            else:
                x = x.transpose(1,2)
                x = self.batchnorm_bb(x)
                weight = self.mlp(x.reshape(b, -1))
                weight = torch.softmax(weight, dim=-1)
                x = torch.mul(x, weight.unsqueeze(1))
                return x.transpose(1,2)
        elif self.optimizer_method == 'normal':
            x = x.transpose(1,2)
            x = self.batchnorm_bb(x)
            weight = self.mlp_bb(x.reshape(b, -1))
            weight = torch.softmax(weight, dim=-1)
            x = torch.mul(x, weight.unsqueeze(1))
            return x.transpose(1,2)