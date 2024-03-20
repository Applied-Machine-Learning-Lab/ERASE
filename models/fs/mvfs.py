import torch
import torch.nn as nn
from models.layers import MLP

class mvfs(nn.Module):

    def __init__(self, args, unique_values, features):
        super(mvfs, self).__init__()

        self.pretrain_epoch = args.fs_config[args.fs]['pretrain_epoch']
        self.feature_num = len(unique_values)
        self.sub_network_num = args.fs_config[args.fs]['sub_network_num']
        self.dropout = args.fs_config[args.fs]['dropout']
        self.l = args.fs_config[args.fs]['l']
        self.sub_network_list_bb = torch.nn.ModuleList()
        for i in range(self.sub_network_num):
            self.sub_network_list_bb.append(nn.Linear(self.feature_num*args.embedding_dim, self.feature_num))
        self.W_g_bb = nn.Parameter(torch.Tensor(self.sub_network_num, self.sub_network_num * self.feature_num))
        self.b_g_bb = nn.Parameter(torch.Tensor(self.sub_network_num))

        self.W = nn.Parameter(torch.Tensor(self.sub_network_num))
        
        self.mode = 'train'
        if self.mode == 'retrain':
            raise Exception('adafs should not be used in retrain mode')
        self.optimizer_method = 'normal'

        self.load_checkpoint = False
        self.t = 1
    
    def forward(self, x, current_epoch, current_step, raw_data):
        b,f,e = x.shape
        if current_epoch is not None and current_epoch < self.pretrain_epoch:
            return x
        else:
            self.t += 0.001
            C = []
            for i in range(self.sub_network_num):
                C.append(torch.softmax(self.sub_network_list_bb[i](x.reshape(b, -1)), dim=1))
            r = self.W_g_bb.unsqueeze(0) @ torch.cat(C, dim=1).reshape(b,-1,1) + self.b_g_bb.reshape(1,-1,1)
            r = torch.softmax(r, dim=1) # b, K, 1
            I = torch.mul(r, torch.stack(C, dim=1))
            I = I.sum(dim=1) # b, f
            # 𝑠𝑛 = 0.5 ∗ (1 + tanh(𝜏 · (𝐼𝑛 − 𝑙)))
            if self.t < 5:
                s = 0.5 * (1 + torch.tanh(5 * (I - self.l)))
            else:
                s = 0.5 * (1 + torch.tanh(self.t * (I-self.l)))
            x = x * s.unsqueeze(2)
            return x
            
