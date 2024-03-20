import torch
import torch.nn as nn

class optfs(nn.Module):

    def __init__(self, args, unique_values, features):
        super().__init__()

        self.gate = {features[field_idx]: torch.Tensor(unique_values[field_idx], 1).to(args.device) for field_idx in range(len(features))}
        for feature in features:
            torch.nn.init.xavier_uniform_(self.gate[feature].data)
        
        self.raw_gate = {features[field_idx]: self.gate[features[field_idx]].clone().detach().to(args.device) for field_idx in range(len(features))}
        self.raw_gc = torch.concat([self.raw_gate[feature] for feature in features], dim=0)

        self.g = {feature: torch.ones_like(self.gate[feature]).to(args.device) for feature in features}
        self.gate = {feature: nn.Parameter(self.gate[feature], requires_grad=True) for feature in features}
        self.gate = torch.nn.ParameterDict(self.gate)

        self.mode = 'train'
        self.device = args.device
        self.features = features
        
        self.gamma = args.fs_config[args.fs]['gamma']
        self.pretrain_epoch = args.fs_config[args.fs]['pretrain_epoch']
        self.load_checkpoint = True
        self.optimizer_method = 'normal'

    def forward(self, x, current_epoch, current_step, raw_data):
        b,f,e = x.shape
        gc = torch.concat([self.gate[feature] for feature in self.features], dim=0)
        if current_epoch is not None: # that's mean, in training or validation
            t = self.gamma ** (current_epoch / self.pretrain_epoch)
        else: # current_epoch is None, that's mean, in test or retrain
            t = self.gamma
        if self.mode == 'train':
            self.g_tmp = torch.sigmoid(gc * t) / torch.sigmoid(self.raw_gc)
            # g_tmp分段赋值给g
            for feature in self.features:
                self.g[feature] = self.g_tmp[:len(self.gate[feature])]
                self.g_tmp = self.g_tmp[len(self.gate[feature]):]
            x_ = torch.zeros_like(x).to(self.device)
            for j in range(f):
                feature = self.features[j]
                x_[:,j,:] = x[:,j,:] * self.g[feature][raw_data[:,j]]
        elif self.mode == 'retrain':
            # self.g_tmp = torch.concat([self.g[feature] for feature in self.features], dim=0)
            x_ = torch.zeros_like(x).to(self.device)
            for j in range(f):
                feature = self.features[j]
                x_[:,j,:] = x[:,j,:] * self.g[feature][raw_data[:,j]]
        
        

        # for i in range(b):
        #     for j in range(f):
        #         feature = self.features[j]
        #         x_[i,j,:] = x[i,j,:] * self.g[feature][raw_data[i,j]]
        # for j in range(f):
        #     feature = self.features[j]
        #     x_[:,j,:] = x[:,j,:] * self.g[feature][raw_data[:,j]]

        return x_
    
    def before_retrain(self):
        # self.gate <= 0 的赋值为0, else 1
        self.gate.requires_grad_(False)
        for feature in self.features:
            self.gate[feature][self.gate[feature] <= 0] = 0
            self.gate[feature][self.gate[feature] > 0] = 1
            print('feature:', feature, 'keep ratio:', torch.sum(self.gate[feature])/self.gate[feature].shape[0])
        self.g = {feature: nn.Parameter(self.gate[feature].clone().detach().to(self.device)) for feature in self.features}
        self.g = torch.nn.ParameterDict(self.g)
        self.g.requires_grad_(False)
