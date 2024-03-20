import random
import numpy as np
import torch
import os
import copy
import importlib
import datetime

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name: str, model_type: str):
    """
    Automatically select model class based on model name

    Args:
        model_name (str): model name
        model_type (str): rec, fs, es

    Returns:
        Recommender: model class
        Dict: model configuration dict
    """
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['models', model_type, model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    else:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    # dir = os.path.dirname(model_module.__file__)
    # conf = dict()
    # fname = os.path.join(os.path.dirname(dir), 'basemodel', 'basemodel.yaml')
    # conf.update(parser_yaml(fname))
    # for name in ['all', model_file_name]:
    #     fname = os.path.join(dir, 'config', name+'.yaml')
    #     if os.path.isfile(fname):
    #         conf = deep_update(conf, parser_yaml(fname))
    return model_class
    

class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
        
    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True

def machine_learning_selection(args, fs, features, unique_values, data, k):
    train_x, train_y, val_x, val_y, test_x, test_y = data
    features = np.array(features)
    if fs == 'lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso(
            alpha=args.fs_config[args.fs]['alpha'],
            fit_intercept=args.fs_config[args.fs]['fit_intercept'],
            copy_X=args.fs_config[args.fs]['copy_X'],
            max_iter=args.fs_config[args.fs]['max_iter'],
            tol=args.fs_config[args.fs]['tol'],
            positive=args.fs_config[args.fs]['positive'],
            selection=args.fs_config[args.fs]['selection']
        )
        lasso.fit(train_x, train_y)
        field_importance = abs(lasso.coef_)
        rank = field_importance.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance[rank]
        return np.array([ranked_features, ranked_importance])
        select_idx = []
        for i in range(k):
            print(features[rank[i]], field_importance[rank[i]])
            select_idx.append(rank[i])
        return features[select_idx]
    elif fs == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier
        gbdt = GradientBoostingClassifier(
            learning_rate=args.fs_config[args.fs]['learning_rate'],
            n_estimators=args.fs_config[args.fs]['n_estimators'],
            subsample=args.fs_config[args.fs]['subsample'],
            min_samples_split=args.fs_config[args.fs]['min_samples_split'],
            min_samples_leaf=args.fs_config[args.fs]['min_samples_leaf'],
            min_weight_fraction_leaf=args.fs_config[args.fs]['min_weight_fraction_leaf'],
            max_depth=args.fs_config[args.fs]['max_depth'],
            n_iter_no_change=args.fs_config[args.fs]['n_iter_no_change'],
            verbose=1
        )
        gbdt.fit(train_x, train_y)
        field_importance = gbdt.feature_importances_
        rank = field_importance.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance[rank]
        return np.array([ranked_features, ranked_importance])
        select_idx = []
        for i in range(k):
            print(features[rank[i]], field_importance[rank[i]])
            select_idx.append(rank[i])
        return features[select_idx]
    elif fs == 'gbr':
        from sklearn.ensemble import GradientBoostingRegressor
        gbr = GradientBoostingRegressor()
        gbr.fit(train_x, train_y)
        field_importance = gbr.feature_importances_
        rank = field_importance.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance[rank]
        return np.array([ranked_features, ranked_importance])
        select_idx = []
        for i in range(k):
            print(features[rank[i]], field_importance[rank[i]])
            select_idx.append(rank[i])
        return features[select_idx]
    elif fs == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k)
        pca.fit(train_x)
        # first component
        field_importance = abs(pca.components_[0])
        rank = field_importance.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance[rank]
        return np.array([ranked_features, ranked_importance])
        select_idx = []
        for i in range(k):
            print(features[rank[i]], field_importance[rank[i]])
            select_idx.append(rank[i])
        return features[select_idx]
    elif fs == 'permutation':
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import Ridge
        from sklearn.inspection import permutation_importance
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, max_depth=None, n_jobs=6, verbose=1).fit(train_x, train_y)
        # model = LogisticRegression(verbose=1,multi_class='ovr',n_jobs=32).fit(train_x, train_y)
        # model = MLPClassifier(verbose=True, early_stopping=True, n_iter_no_change=3, hidden_layer_sizes=(16,16)).fit(train_x, train_y)
        field_importance = permutation_importance(model, train_x, train_y, n_jobs=5)
        rank = field_importance.importances_mean.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance.importances_mean[rank]
        return np.array([ranked_features, ranked_importance])
    elif fs == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, max_depth=None, n_jobs=6, verbose=1).fit(train_x, train_y)
        field_importance = model.feature_importances_
        rank = field_importance.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance[rank]
        return np.array([ranked_features, ranked_importance])
    elif fs == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=10, max_depth=None, n_jobs=6, verbose=1).fit(train_x, train_y)
        field_importance = model.feature_importances_
        rank = field_importance.argsort()[::-1]
        ranked_features = features[rank]
        ranked_importance = field_importance[rank]
        return np.array([ranked_features, ranked_importance])

    
def print_time(message):
    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S '), message)

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')