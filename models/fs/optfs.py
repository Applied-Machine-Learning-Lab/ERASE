import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class optfs(nn.Module):

    def __init__(self, args, unique_values, features):
        super().__init__()

        self.mask_weight = nn.Parameter(torch.Tensor(np.sum(unique_values), 1))
        nn.init.constant_(self.mask_weight, 0.5)
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))

        self.mode = 'train'
        self.device = args.device
        self.features = features
        
        self.gamma = args.fs_config[args.fs]['gamma']
        if args.dataset == 'avazu':
            self.gamma = 5000
        elif args.dataset == 'criteo':
            self.gamma = 2000
        else:
            self.gamma = 2000
        self.pretrain_epoch = args.fs_config[args.fs]['pretrain_epoch']
        self.load_checkpoint = True
        self.optimizer_method = 'normal'

        self.temp_increase = self.gamma ** (1./ (self.pretrain_epoch-1))
        self.temp = 1
        self.current_epoch = -1
    
    def sigmoid(self, x):
        return float(1./(1.+np.exp(-x)))

    def compute_mask(self, raw_data, temp, ticket):
        scaling = 1./ self.sigmoid(0.5)
        mask_weight = F.embedding(raw_data + raw_data.new_tensor(self.offsets), self.mask_weight)
        if ticket:
            mask = (mask_weight > 0).float()
        else:
            mask = torch.sigmoid(temp * mask_weight)
        return scaling * mask

    def forward(self, x, current_epoch, current_step, raw_data):
        b,f,e = x.shape
        if current_epoch != self.current_epoch:
            self.temp *= self.temp_increase
            self.current_epoch = current_epoch
        if self.mode == 'retrain': 
            ticket = True
        else: 
            ticket = False
        mask = self.compute_mask(raw_data, self.temp, ticket)

        return x * mask
    
    def before_retrain(self):
        # print remain
        ratio = float((self.mask_weight > 0).sum()) / self.mask_weight.numel()
        print('remain: ', ratio)
