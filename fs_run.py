import sys
import pandas as pd
import numpy as np
import torch
import os
import argparse
import yaml
import nni
import time
import datetime as dt
from tqdm import tqdm
import utils.utils as utils
from utils.fs_trainer import modeltrainer
from utils.datasets import read_dataset
from models.basemodel import BaseModel


def main(args):
    if args.seed != 0:
        utils.seed_everything(args.seed)
    
    if args.train_or_search:
        utils.print_time('start train or search...')

        if args.fs in ['gbdt', 'lasso', 'permutation','rf','xgb']: # machine learning feature selection
            features, unique_values, data = read_dataset(args.dataset, args.data_path, args.batch_size, args.dataset_shuffle, num_workers=args.num_workers, machine_learning_method = True)
            ml_start_time = dt.datetime.now()
            feature_rank = utils.machine_learning_selection(args, args.fs, features, unique_values, data, args.k)
            ml_end_time = dt.datetime.now()
            print('machine learning feature selection time: {} s'.format((ml_end_time - ml_start_time).total_seconds()))
            model_path = 'checkpoints/' + args.model + '_' + args.fs + '_' + args.es + '_' + args.dataset + '_' + args.timestr + '/'
            utils.print_time(model_path)
            utils.print_time('feature rank:')
            utils.print_time(feature_rank)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            np.save(model_path + 'feature_rank.npy', feature_rank)

        else:
            features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_dataset(args.dataset, args.data_path, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers, machine_learning_method=False)

            print(features)
            print(unique_values)

            model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
            model.fs.mode = 'train'
            trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=False)
            trainer.fit(train_dataloader, val_dataloader)
            auc = trainer.test(test_dataloader, ['auc', 'logloss'])
            if args.retrain is False and args.nni:
                nni.report_final_result(auc)
            # print selected features
            # 如果model存在output_selected_features方法，就输出selected features
            if hasattr(model.fs, 'save_selection'):
                model_path = 'checkpoints/' + args.model + '_' + args.fs + '_' + args.es + '_' + args.dataset + '_' + args.timestr + '/'
                res = model.fs.save_selection(k=args.k)
                if isinstance(res, np.ndarray):
                    feature_rank = res
                else: 
                    feature_rank = res(val_dataloader,model,torch.device(args.device))
                utils.print_time('feature rank:')
                utils.print_time(feature_rank)
                np.save(model_path + 'feature_rank.npy', feature_rank)
    # if retrain, retrain
    if args.retrain:
        utils.print_time('start retrain...')
        # if no need train or search, you should put the feature_rank.npy in the following path mannualy
        if not args.train_or_search:
            model_path = 'checkpoints_for_retrain/' + args.rank_path + '/'
            if not os.path.exists(model_path):
                raise('Only retraining chossen, please make shure you have putted the generated file during searching in the following path: checkpoints/for_retrain/fs_es_dataset/')
        else:
            model_path = 'checkpoints/' + args.model + '_' + args.fs + '_' + args.es + '_' + args.dataset + '_' + args.timestr + '/'
        # read selection results
        if args.read_feature_rank: # need to read selected features
            feature_rank = np.load(model_path + 'feature_rank.npy')
            selected_features = feature_rank[0][:args.k]
            utils.print_time('feature rank: {}'.format(feature_rank))
            utils.print_time('selected features: {}'.format(selected_features))
            features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_dataset(args.dataset, args.data_path, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers, use_fields=selected_features, machine_learning_method=False)
        else:
            features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_dataset(args.dataset, args.data_path, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers, use_fields=None, machine_learning_method=False)
        
        model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        if model.fs.load_checkpoint:
            model.load_state_dict(torch.load(model_path + 'model_search.pth'))
        # if args.fs == 'optfs':
        #     tmp_model = torch.load(model_path + 'model_search.pth')
        #     # param_dict = {k:v for k, v in tmp_model.items() if 'fs.gate' in k or 'embedding.weight' in k}
        #     param_dict = {k:v for k, v in tmp_model.items() if 'fs.mask_weight' in k}
        #     model_dict = model.state_dict()
        #     model_dict.update(param_dict)
        #     model.load_state_dict(model_dict)
        if hasattr(model.fs, 'before_retrain'):
            model.fs.before_retrain()
        model.fs.mode = 'retrain'
        
        trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=True)
        trainer.fit(train_dataloader, val_dataloader)
        auc = trainer.test(test_dataloader, ['auc', 'logloss'])
        if args.nni:
            nni.report_final_result(auc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avazu', help='avazu, criteo, movielens-1m, aliccp')
    parser.add_argument('--model', type=str, default='mlp', help='mlp, ...')
    parser.add_argument('--fs', type=str, default='no_selection', help='feature selection methods: no_selecion, autofield, adafs, optfs, gbdt, lasso, gbr, pca, shark, sfs, lpfs, mvfs')
    parser.add_argument('--es', type=str, default='no_selection', help='embedding search methods: no_selecion, ...')
    parser.add_argument('--seed', type=int, default=0, help='random seed, 0 represents not setting the random seed')
    parser.add_argument('--device',type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu, cuda')
    parser.add_argument('--data_path', type=str, default='data/', help='data path') # ~/autodl-tmp/ or data/
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--dataset_shuffle', type=bool, default=True, help='whether to shuffle the dataset')
    parser.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension')
    parser.add_argument('--train_or_search', type=utils.str2bool, default=True, help='whether to train or search')
    parser.add_argument('--retrain', type=utils.str2bool, default=True, help='whether to retrain')
    parser.add_argument('--k', type=int, default=0, help='top k features')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='epoch')
    parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=32, help='num_workers')
    parser.add_argument('--nni', type=bool, default=False, help='whether to use nni')
    parser.add_argument('--rank_path', type=str, default='None', help='if only retrain, no train, please specify the path of feature_rank file. e.g., autofield_no_selection_avazu')
    parser.add_argument('--read_feature_rank', type=utils.str2bool, default=True, help='whether to use pre-saved feature rank')

    args = parser.parse_args()
    
    # k
    if args.k == 0:
        if args.dataset == 'avazu':
            args.k = 22
        elif args.dataset == 'criteo':
            args.k = 39

    with open('models/config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    args.__dict__.update(data)

    # read tune parameters from nni
    if args.nni:
        tuner_params = nni.get_next_parameter()
        for key in tuner_params:
            if key[:2] == 'fs':
                args.fs_config[args.fs][key[3:]] = tuner_params[key]
            elif key[:2] == 'es':
                args.es_config[args.es][key[3:]] = tuner_params[key]
            else:
                args.__dict__[key] = tuner_params[key]
    
    # print args
    for key in args.__dict__:
        if key not in ['fs_config', 'es_config', 'rec_config']:
            print(key, ':', args.__dict__[key])
        else:
            print(key, ':')
            for key2 in args.__dict__[key]:
                if key2 in [args.model, args.fs, args.es]:
                    print('\t', key2, ':', args.__dict__[key][key2])
    
    args.timestr = str(time.time())
    
    main(args)