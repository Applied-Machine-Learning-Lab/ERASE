import torch
import torch.nn as nn
import numpy as np

class lpfs(nn.Module):
    def __init__(self, args, unique_values, features):
        super(lpfs, self).__init__()
        
        self.feature_num = len(features)
        self.features = features
        self.x = nn.Parameter(torch.ones(self.feature_num, 1).to(args.device))
        self.epochs = args.epoch
        self.epsilon_update_frequency = 100
        self.device = args.device
        self.epsilon = 0.1

        self.load_checkpoint = False
        self.optimizer_method = 'normal'

    def forward(self, x, current_epoch, current_step, raw_data):
        b,f,e = x.shape
        if current_step % self.epsilon_update_frequency == 0:
            self.epsilon = self.epsilon * 0.9978
        g = self.lpfs_pp(self.x, self.epsilon).reshape(1, f, 1)
        x_ = torch.zeros_like(x)
        x_ = x * g
        return x_
    
    def lpfs_pp(self, x, epsilon, alpha=10, tau=2, init_val=1.0):
        g1 = x*x/(x*x+epsilon)
        g2 = alpha * epsilon ** (1.0/tau)*torch.atan(x)
        g = torch.where(x>0, g2+g1, g2-g1)/init_val
        return g
    
    def save_selection(self, k):
        # gate = torch.concat([self.gate[self.features[field_idx]] for field_idx in range(self.feature_num)], dim=0)[:,-1]
        gate = self.x.reshape(self.feature_num)
        indices = torch.argsort(gate, descending=True)
        ranked_importance = gate[indices].detach().cpu().numpy()
        ranked_features = [self.features[i] for i in indices]
        print(ranked_features)
        print(ranked_importance)
        return np.array([ranked_features, ranked_importance])

