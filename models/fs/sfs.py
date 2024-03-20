import torch
import torch.nn as nn
import numpy as np
import tqdm
import numpy as np

class sfs(nn.Module):

    def __init__(self, args, unique_values, features):
        super(sfs, self).__init__()
        self.load_checkpoint = False
        self.optimizer_method = 'normal'

        self.feature_num = len(unique_values)
        self.device = args.device
        self.args = args
        self.criterion = torch.nn.BCELoss()
        self.features = np.array(features)    

        opt = args.fs_config[args.fs]
        #self.cr = opt['cr']
        self.num_batch_sampling = opt['num_batch_sampling']

        self.mode = 'train'
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))
        print(self.offsets)
        print(self.feature_num)
        self.mask = nn.Parameter(torch.ones([self.feature_num,1]))
        self.mask.requires_grad = False
    
    def forward(self, x, current_epoch, current_step, raw_data):
        return x*self.mask
    
    def save_selection(self, k):
        def prun(dataloader,model,device):
            model.fs.mask.requires_grad = True
            for i, (c_data, labels) in enumerate(dataloader):
                if i == model.fs.num_batch_sampling:
                    break
                c_data, labels = c_data.to(device), labels.to(device)
                out = model(c_data,0,i)
                loss =self.criterion(out, labels.float().unsqueeze(-1))
                model.zero_grad()
                loss.backward()            
                grads = torch.abs(model.fs.mask.grad)
                if i == 0:
                    moving_average_grad = grads
                else:
                    moving_average_grad =  ((moving_average_grad * i) + grads) / (i + 1)
            grads = torch.flatten(moving_average_grad)
            importance = grads / grads.sum()
            feature_rank = torch.argsort(importance, descending=True)
            ranked_importance = importance[feature_rank].detach().cpu().numpy()
            ranked_features = [self.features[i] for i in feature_rank]
            return np.array([ranked_features, ranked_importance])
        return prun
       

    


