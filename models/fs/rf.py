import torch
import torch.nn as nn
import torch.nn.functional as F

class rf(nn.Module):
    def __init__(self, args, unique_values, features):
        super(rf, self).__init__()

        # 必需的参数
        self.load_checkpoint = False
        self.optimizer_method = 'normal'

    def forward(self, x, current_epoch, current_step, raw_data):
        return x