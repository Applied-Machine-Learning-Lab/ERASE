import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

class shark(nn.Module):
    def __init__(self, args, unique_values, features):
        super(shark, self).__init__()
        self.feature_num = len(unique_values)
        self.features = np.array(features)
        # 必需的参数
        self.load_checkpoint = False
        self.optimizer_method = 'normal'
        self.criterion = torch.nn.BCELoss()
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))

    def forward(self, x, current_epoch, current_step, raw_data):
        return x
    
    def save_selection(self, k):
        def selection(test_dataloader, model, device):
            tk0 = tqdm.tqdm(test_dataloader, desc="f-permutation", smoothing=0, mininterval=1.0)
            model = model.to(device)
            num = 0
            # importance = torch.zeros(len(model.offsets)).to(device) # save importance for each field
            importance = np.zeros(len(model.offsets))
            expectation = torch.zeros((len(model.offsets))).to(device)
            for x,y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                embs = model.embedding(x + x.new_tensor(self.offsets))
                if len(expectation.shape) == 1:
                    expectation = torch.zeros((len(model.offsets), embs.shape[2])).to(device)
                expectation += torch.sum(embs, dim=0)
                num += x.shape[0]
            expectation = expectation / num
            expectation = expectation.reshape(1, len(model.offsets), -1)
            # expectation = torch.zeros((1, len(model.offsets), 8)).to(device)
            num = 0
            new_dataloader = torch.utils.data.DataLoader(test_dataloader.dataset, batch_size=1, num_workers=16)
            tk0 = tqdm.tqdm(new_dataloader, desc="f-permutation", smoothing=0, mininterval=1.0)
            for i, (x, y) in enumerate(tk0):
                x = x.to(device)
                y = y.to(device)
                model.zero_grad()
                embs = model.embedding(x + x.new_tensor(self.offsets))
                # expectation = torch.mean(embs, dim=0)
                expectation_resize = expectation.repeat(x.shape[0], 1,1)
                right_part = expectation_resize - embs
                y_pred = model(x, current_epoch=None, current_step=i)
                loss = self.criterion(y_pred, y.float().reshape(-1, 1))
                # cal gradient for each embedding
                loss.backward()
                # get gradient
                gradients = F.embedding(x + x.new_tensor(self.offsets),model.embedding.weight.grad).to(device)
                # use the torch.gradient
                # cal importance
                error = gradients * right_part # b,f,e
                error = torch.sum(error, dim=2) # b,f
                error = torch.sum(abs(error), dim=0) # f
                importance += error.detach().cpu().numpy()
                num += x.shape[0]
            importance = importance / num
            # sort importance
            feature_rank = np.argsort(importance)[::-1]
            ranked_importance = importance[feature_rank]
            ranked_features = [self.features[i] for i in feature_rank]
            return np.array([ranked_features, ranked_importance])
        return selection