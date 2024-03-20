import torch
import torch.nn as nn
import tqdm
import os
import nni
import datetime as dt
from utils.utils import EarlyStopper
from sklearn.metrics import roc_auc_score, log_loss

class modeltrainer():
    def __init__(self, args, model, model_name, device, epochs, retrain):
        self.args = args
        self.model = model
        self.optimizers = model.set_optimizer() # dict of optimizers
        self.criterion = torch.nn.BCELoss()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.n_epoch = epochs
        self.model_path = 'checkpoints/' + model_name + '_' + args.fs + '_' + args.es + '_' + args.dataset + '_' + args.timestr + '/'
        self.early_stopper = EarlyStopper(patience=args.patience)
        self.retrain = retrain

    def train_one_epoch(self, train_dataloader, val_dataloader, epoch_i, log_interval=10):
        self.model.train()
        total_loss = 0
        val_iter = iter(val_dataloader)
        tk0 = tqdm.tqdm(train_dataloader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x, y) in enumerate(tk0):
            # x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x, current_epoch=epoch_i, current_step=i)
            loss = self.criterion(y_pred, y.float().reshape(-1, 1))
            # optfs l1 norm
            if self.args.fs == 'optfs' and not self.retrain:
                reg_loss = torch.sum(torch.sigmoid(self.model.fs.temp * self.model.fs.mask_weight))
                # g = torch.concat([self.model.fs.g[feature] for feature in self.model.fs.features], dim=0)
                # l1_loss = torch.norm(g, p=1) * 2e-9
                if self.args.dataset == 'avazu':
                    loss = loss + reg_loss * 4e-9
                elif self.args.dataset == 'criteo':
                    loss = loss + reg_loss * 1e-8
                elif self.args.dataset == 'movielens-1m':
                    loss = loss + reg_loss * 1e-4
                elif self.args.dataset == 'aliccp':
                    loss = loss + reg_loss * 1e-8
                else:
                    print('please set the hyparameters for optfs of reg_loss in fs_trainer.py')

            self.model.zero_grad()
            # self.optimizer.zero_grad()
            loss.backward()
            self.optimizers['optimizer_bb'].step()
            if self.args.fs == 'optfs' and not self.retrain:
                self.optimizers['optimizer_fs'].step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

            # other optimizers
            if self.model.fs.optimizer_method == 'darts' and i % self.model.fs.update_frequency == 0:
                self.optimizers['optimizer_fs'].zero_grad()
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    batch = next(val_iter)
                x_,y_ = batch
                x_, y_ = x_.to(self.device), y_.to(self.device)
                y_pred_ = self.model(x_, current_epoch=epoch_i, current_step=i)
                loss_ = self.criterion(y_pred_, y_.float().reshape(-1, 1))
                loss_.backward()
                self.optimizers['optimizer_fs'].step()
            elif self.args.fs == 'lpfs':
                p = self.optimizers['optimizer_fs'].param_groups[0]['params'][0]
                self.optimizers['optimizer_fs'].step()
                thr = 0.01 * self.args.learning_rate
                in1 = p.data > thr
                in2 = p.data < -thr
                in3 = ~(in1 | in2)
                p.data[in1] -= thr
                p.data[in2] += thr
                p.data[in3] = 0.0

            

    def fit(self, train_dataloader, val_dataloader=None):
        all_start_time = dt.datetime.now()
        epoch_time_lis = []
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            epoch_start_time = dt.datetime.now()
            self.train_one_epoch(train_dataloader, val_dataloader, epoch_i)
            epoch_end_time = dt.datetime.now()
            epoch_time_lis.append((epoch_end_time - epoch_start_time).total_seconds())
            if val_dataloader:
                auc = self.evaluate(val_dataloader, epoch_i)
                # nni
                if self.args.nni:
                    nni.report_intermediate_result(auc.item())
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
                # stop early stopper during adafs pretrain
                if self.args.fs in ['adafs','mvfs'] and epoch_i < self.model.fs.pretrain_epoch:
                    print('reset early stopper due to pretraining')
                    self.early_stopper.trial_counter = 0
                    self.early_stopper.best_auc = 0
                    self.early_stopper.best_weights = None
        all_end_time = dt.datetime.now()
        print('all training time: {} s'.format((all_end_time - all_start_time).total_seconds()))
        print('average epoch time: {} s'.format(sum(epoch_time_lis) / len(epoch_time_lis)))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if self.model.fs.mode != 'retrain':
            torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_search.pth"))  #save best auc model
        # else:
        #     torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_retrain.pth"))
    
    def evaluate(self, data_loader, current_epoch):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x, y) in enumerate(tk0):
                x = x.to(self.device)
                # x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = self.model(x, current_epoch, current_step=i) # current_epoch=None means not in training mode
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return roc_auc_score(targets, predicts)
    
    def test(self, data_loader, evaluate_fns):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="test", smoothing=0, mininterval=1.0)
            start_time = dt.datetime.now()
            for i, (x, y) in enumerate(tk0):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x, current_epoch=None, current_step=i)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
            end_time = dt.datetime.now()
            print('infer time: {} s'.format((end_time - start_time).total_seconds()))
        for evaluate_fn in evaluate_fns:
            if evaluate_fn == 'auc':
                auc = roc_auc_score(targets, predicts)
                print('test auc:', auc)
            elif evaluate_fn == 'logloss':
                logloss = log_loss(targets, predicts)
                print('test logloss:', logloss)
        return auc
