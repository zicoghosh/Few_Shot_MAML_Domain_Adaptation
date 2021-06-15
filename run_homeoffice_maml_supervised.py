"""
Train a model on miniImageNet.
"""

import random
import os
import ntpath
import sys
import numpy as np

import torch

import argparse
from functools import partial
from datetime import datetime

project_path = "/home/sankha/Surjayan/"

miniImageNet_pretrained_model_path = '/home/sankha/Few-shot-MO/results_auto/miniimagenetImb_maml_checkpoint_order_2_seed_0_datetime_27_10_2020_18_55_05_shots_5_ways_5/'

DATA_DIR = project_path + "Data/OfficeHomeDataset/"

#Load necessary user defined functions 
sys.path.insert(0,project_path)

from officehome import read_dataset, augment_dataset
from maml_da_src_supervised.train_maml import train

from maml_da_src_supervised.models_maml import MiniImageNetModel
from maml_da_src_supervised.eval_model_maml import evaluate
from maml_da_src_supervised.maml_da import MAML

# print("Successfully loaded user defined functions")

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--order', help='MAML order can be 2 or 1', default=2, type=int)
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=1, type=int)
    parser.add_argument('--meta-shots', help='shots for meta update', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=5, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true', default=False)
    parser.add_argument('--learning-rate',help='base learner update step size', default=0.01, type=float)
    parser.add_argument('--meta-step', help='meta-learner Adam step size', default=0.001, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=4, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=60000, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=10, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=600, type=int)
    parser.add_argument('--eval-interval',help='Evaluation interval during training', default=1000, type=int)
    parser.add_argument('--eval-interval-sample',help='evaluation samples during training', default=100, type=int)
    parser.add_argument('--only-evaluation', help='Set for only evaluation', action='store_true', default=False)
    parser.add_argument('--checkpoint', help='Load saved checkpoint from path', default=None)
    #New Args
    parser.add_argument('--dropout', help='dropout', default=0.0, type=float)
    parser.add_argument('--augment', help='Set for applying rand-augment with prob = 0.5', action='store_true', default=False)
    parser.add_argument('--label_smoothing_enable', help='Set for enabling label-smoothing', action='store_true', default=False)
    parser.add_argument('--pre_train_wts_enable', help='Set for using pre_trained miniimagenet wts', action='store_true', default=False)
    parser.add_argument('--pre_train_checkpoint', help='Load saved pretrain csheckpoint from path', default=None)
    parser.add_argument('--test_domain', help='specify domain to not use during training', default='Product', type=str)
    return parser

def model_kwargs(parsed_args):

    return {
        'update_lr': parsed_args.learning_rate,
        'meta_step_size': parsed_args.meta_step,
        'label_smoothing_enable': parsed_args.label_smoothing_enable
    }

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'order': parsed_args.order,
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'meta_shots': parsed_args.meta_shots,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'num_samples': parsed_args.eval_samples,
        'eval_interval_sample': parsed_args.eval_interval_sample,
        'sample_acc_interval_domain':parsed_args.test_domain
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'num_samples': parsed_args.eval_samples
    }


def getBaseName(class_obj):
    return ntpath.basename(class_obj)


def get_domains_(test_domain: str, domain_list: list):
    temp_domain =''
    if(test_domain == "RealWorld"):
        temp_domain = "Real World"
    else:
        temp_domain = test_domain

    train_domains = []
    test_domain = ''
    for domain in domain_list:
        if(domain == temp_domain):
            test_domain = domain
        else:
            train_domains.append(domain)
    
    return train_domains, test_domain


def main():
    """
     Load data and train a model on it.
    """
    sys.dont_write_bytecode = True

    args = argument_parser().parse_args()
    torch.manual_seed(args.seed)

    """
    train domain : to be used during training
    test domain : unseen domain during training
    """
    train_domains, test_domain = get_domains_(test_domain = args.test_domain, domain_list = ["Art", "Clipart", "Product", "Real World"])


    dt = datetime.now()
    dt_str = dt.strftime("%d_%m_%Y_%H_%M_%S")

    # fileStart = 'officehome_maml_supervised_' + str(args.order) + '_seed_' + str(args.seed) + \
    #         '_datetime_' + dt_str + \
    #         '_shots_' + str(args.shots) + '_ways_' + str(args.classes)

    fileStart = 'officehome_maml_order_' + str(args.order) + '_seed_' + str(args.seed) + '_shots_' + str(args.shots) + \
                '_ways_' + str(args.classes) + '_meta-lr_' + str(args.meta_step) + '_test-domain_' + test_domain

    if args.dropout > 0.0:
        fileStart = fileStart + '_dropout_' + str(args.dropout)
    if args.augment:
        fileStart = fileStart  + '_RandAugment-prob_0.5'
    if args.label_smoothing_enable:
        fileStart = fileStart + '_label-smoothing_0.05'
    if args.pre_train_wts_enable:
        fileStart = fileStart + '_pre-train-wts'

    fileStart = fileStart + '_datetime_' + dt_str

    if args.only_evaluation is False:
        print("configuration : ",fileStart)

    print("\nTrain domains : ",train_domains)
    print("Test domain :", test_domain)

    if args.only_evaluation is False:
        if not os.path.exists('results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart):
            os.makedirs('results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart)

    device = torch.device('cuda')

    train_set, _, test_set = read_dataset(DATA_DIR)
    if args.only_evaluation is False:
        if args.augment:
            train_set = list(augment_dataset(train_set))

    print("\nTrain len", len(train_set))
    print("Test len", len(test_set))

    model=MiniImageNetModel(args.classes)

    maml_model = MAML(model, device, **model_kwargs(args))

    if args.only_evaluation is True:

        assert args.checkpoint is not None, 'For evaluating without training please provide a checkpoint'
        print("Model No:",getBaseName(args.checkpoint))

        checkpoint_model = torch.load(args.checkpoint)
        maml_model.net.load_state_dict(checkpoint_model['model_state'])
        maml_model.meta_optim.load_state_dict(checkpoint_model['meta_optim_state'])

    else:
        if args.pre_train_wts_enable:
            path_model_pt = miniImageNet_pretrained_model_path + 'final_model.pt'
            print("\nUsing pretrained model from path :", path_model_pt)
            print("\n")

            checkpoint__pretrained_model = torch.load(path_model_pt)
            maml_model.net.load_state_dict(checkpoint__pretrained_model['model_state'])

        train(maml_model, train_set, None , test_set, 'results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart, **train_kwargs(args))
        print("Training Complete\n\n")
        
        save_path = 'results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart + '/' + 'final_model.pt'

        torch.save({'model_state': maml_model.net.state_dict(),
                    'meta_optim_state': maml_model.meta_optim.state_dict()},
                    save_path)

    # print('Evaluating...')
    eval_kwargs = evaluate_kwargs(args)

    train_SS_accuracies_sum = 0
    train_SS_accuracy_sum = 0
    train_SS_variation_sum = 0
    print("\nEvaluating for Seen-Domains('Art', 'Clipart', 'Product') and Seen Classes(Train):")
    for eval_domain in ["Art", "Clipart", "Product"]:
        print("\tDomain :",eval_domain)        
        SS_accuracies, SS_accuracy, SS_variation = evaluate(maml_model, train_set, eval_domain = eval_domain, **eval_kwargs)
        print('\tAccuracy: ' + str(SS_accuracy) + '; variation: ' + str(SS_variation))
        print("\n")
        train_SS_accuracies_sum += SS_accuracies
        train_SS_accuracy_sum += SS_accuracy
        train_SS_variation_sum += SS_variation

    train_SS_accuracies_sum = train_SS_accuracies_sum/3

    test_SU_accuracies_sum = 0
    test_SU_accuracy_sum = 0
    test_SU_variation_sum = 0
    print("\nEvaluating for Seen-Domains ",train_domains," and Unseen Classes(Test):")
    for eval_domain in train_domains:
        print("\tDomain :",eval_domain)        
        SU_accuracies, SU_accuracy, SU_variation = evaluate(maml_model, test_set, eval_domain = eval_domain, **eval_kwargs)
        print('\tAccuracy: ' + str(SU_accuracy) + '; variation: ' + str(SU_variation))
        print("\n")
        test_SU_accuracies_sum += SU_accuracies
        test_SU_accuracy_sum += SU_accuracy
        test_SU_variation_sum += SU_variation
        
    test_SU_accuracies_sum = test_SU_accuracies_sum/3

    print("\nEvaluating for Unseen Domain ",test_domain," and Seen Classes:")
    US_accuracies, US_accuracy, US_variation = evaluate(maml_model, train_set, "Real World", **eval_kwargs)
    print('Accuracy: ' + str(US_accuracy) + '; Variation: ' + str(US_variation))

    print("\nEvaluating for Unseen Domain ",test_domain," and Unseen Classes:")
    UU_accuracies, UU_accuracy, UU_variation = evaluate(maml_model, test_set, "Real World", **eval_kwargs)
    print('Accuracy: ' + str(UU_accuracy) + '; Variation: ' + str(UU_variation))

    if args.only_evaluation is False:
        if not os.path.exists('results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart):
            os.makedirs('results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart)
        save_path = 'results_auto/01_domain_adaptation/maml_supervised/June_14/' + fileStart + '/' + 'results.npz'
        np.savez(save_path, UU_accuracies=UU_accuracies, US_accuracies=US_accuracies, SU_accuracies=test_SU_accuracies_sum, SS_accuracies=train_SS_accuracies_sum)

if __name__ == '__main__':
   main()