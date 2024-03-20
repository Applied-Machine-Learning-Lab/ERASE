import nni
import argparse
import json
import os
import re
from nni.experiment import Experiment
from utils.utils import str2bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='criteo', help='avazu, criteo')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--fs', type=str, default='no_selection')
    parser.add_argument('--es', type=str, default='no_selection')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--train_or_search', type=str2bool, default=True, help='whether to train or search')
    parser.add_argument('--retrain', type=str2bool, default=True, help='whether to retrain')
    parser.add_argument('--k', type=int, default=0, help='top k features, if set, use just this k')
    parser.add_argument('--port', type=int, default=8080, help='port of nni server')

    args = parser.parse_args()
    script_name = None
    if args.es != 'no_selection':
        script_name = 'es_run.py'
    else:
        script_name = 'fs_run.py'

    field_num = 0
    if args.dataset == 'avazu':
        field_num = 22
    elif args.dataset == 'criteo':
        field_num = 39

    fs_search_space, es_search_space, model_search_space = None, None, None
    fs_search_space_path = 'nni/search_spaces/fs/' + args.fs + '.json'
    es_search_space_path = 'nni/search_spaces/es/' + args.es + '.json'
    model_search_space_path = 'nni/search_spaces/config.json'
    if not os.path.exists(fs_search_space_path):
        print('fs search space not exists, skip')
    else:
        with open(fs_search_space_path, 'r') as f:
            fs_search_space = json.load(f)
    if not os.path.exists(es_search_space_path):
        print('es search space not exists, skip')
    else:
        with open(es_search_space_path, 'r') as f:
            es_search_space = json.load(f)
    with open(model_search_space_path, 'r') as f:
        model_search_space = json.load(f)
    search_space = {}
    if fs_search_space is not None:
        search_space.update(fs_search_space)
    if es_search_space is not None:
        search_space.update(es_search_space)
    search_space.update(model_search_space)

    if args.k == 0:
        # if no specific k, set k to be a random value between field_num * 0.8 and field_num
        search_space["k"] = {"_type": "randint", "_value": [int(field_num * 0.8), field_num]}
    else:
        # if specific k, set k to be a choice value
        search_space["k"] = {"_type": "choice", "_value": [args.k]}
    
    experiment = Experiment('local')
    experiment.config.experiment_name = args.dataset + '_' + args.model + '_' + args.fs + '_' + args.es
    experiment.config.trial_command = 'python {} --dataset={} --model={} --fs={} --es={} --data_path={} --nni=True --train_or_search={} --retrain={} --k={}'.format(script_name, args.dataset, args.model, args.fs, args.es, args.data_path, args.train_or_search, args.retrain, args.k)
    experiment.config.trial_code_directory = '.' # code directory
    experiment.config.experiment_working_directory = 'experiments/' # working directory
    if not os.path.exists(experiment.config.experiment_working_directory):
        os.makedirs(experiment.config.experiment_working_directory)
    experiment.config.search_space = search_space

    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    experiment.config.max_trial_number = 16
    experiment.config.trial_concurrency = 8
    experiment.config.max_experiment_duration = '24h'

    experiment.config.trial_gpu_number = 1
    experiment.config.training_service.use_active_gpu = True

    experiment.run(args.port)
    # experiment_id = nni.get_experiment_id()
    # # get the best parameters
    # experiment_dir = os.path.join('nni-experiments',experiment_id, 'trials')
    # auc_value, logloss_value = 0.0, 100.0
    # best_trial = None
    # for trial in os.listdir(experiment_dir):
    #     file_path = os.path.join(experiment_dir, trial, 'trial.log')
    #     auc_pattern = r"test auc: ([0-9.]+)"
    #     logloss_pattern = r"test logloss: ([0-9.]+)"
    #     with open(file_path, "r") as file:
    #         lines = file.readlines()
    #         auc_match = re.search(auc_pattern, lines[-2])
    #         logloss_match = re.search(logloss_pattern, lines[-1])
    #         if auc_match:
    #             auc_value = max(auc_value, float(auc_match.group(1)))
    #             if auc_value == float(auc_match.group(1)):
    #                 best_trial = trial
    #         if logloss_match:
    #             logloss_value = min(logloss_value, float(logloss_match.group(1)))
    # print('best trial: ', best_trial)
    # print('best auc: ', auc_value)
    # print('best logloss: ', logloss_value)
    # print('best parameters:')
    # best_trial_para_path = os.path.join(experiment_dir, best_trial, 'parameter.cfg')
    # with open(best_trial_para_path, 'r') as file:
    #     lines = file.readlines()
    #     print(lines)
    
    experiment.stop()
