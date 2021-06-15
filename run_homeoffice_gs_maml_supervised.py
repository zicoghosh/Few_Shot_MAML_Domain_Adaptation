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

project_path = "/home/sankha/Surjayan/Few_Shot_learning_Domain_Adaptation/"

DATA_DIR = project_path + "Data/OfficeHomeDataset/"

#Load necessary user defined functions 
sys.path.insert(0,project_path)

from officehome import read_dataset
from maml_da_src_supervised.train_maml import train

from maml_da_src_supervised.models_maml import MiniImageNetModel
from maml_da_src_supervised.eval_model_maml import evaluate
from maml_da_src_supervised.gs_maml_da import MAML

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
    return parser

def model_kwargs(parsed_args):

    return {
        'update_lr': parsed_args.learning_rate,
        'meta_step_size': parsed_args.meta_step
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
        'eval_interval_sample': parsed_args.eval_interval_sample
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

def getModelName(class_obj):
    return ntpath.basename(class_obj)

def main():
    """
    Load data and train a model on it.
    """
    sys.dont_write_bytecode = True
    
    args = argument_parser().parse_args()
    torch.manual_seed(args.seed)

    dt = datetime.now()
    dt_str = dt.strftime("%d_%m_%Y_%H_%M_%S")

    fileStart = 'officehome_gs_maml_checkpoint_order_' + str(args.order) + '_seed_' + str(args.seed) + \
            '_datetime_' + dt_str + \
            '_shots_' + str(args.shots) + '_ways_' + str(args.classes)
    
    if args.only_evaluation is False:
        print(fileStart)

    if args.only_evaluation is False:
        if not os.path.exists('results_auto/domain_adaptation/gs_maml_supervised/' + fileStart):
            os.makedirs('results_auto/domain_adaptation/gs_maml_supervised/' + fileStart)

    device = torch.device('cuda')

    train_set, _, test_set = read_dataset(DATA_DIR)

    model=MiniImageNetModel(args.classes)

    maml_model = MAML(model, device, **model_kwargs(args))

    if args.only_evaluation is True:

        assert args.checkpoint is not None, 'For evaluating without training please provide a checkpoint'
        print("Model No:",getModelName(args.checkpoint))

        checkpoint_model = torch.load(args.checkpoint)
        maml_model.net.load_state_dict(checkpoint_model['model_state'])
        maml_model.meta_optim.load_state_dict(checkpoint_model['meta_optim_state'])

    else:
        train(maml_model, train_set, None , test_set, 'results_auto/domain_adaptation/gs_maml_supervised/' + fileStart, **train_kwargs(args))
        print("Training Complete\n\n")
        
        save_path = 'results_auto/domain_adaptation/gs_maml_supervised/' + fileStart + '/' + 'final_model.pt'

        torch.save({'model_state': maml_model.net.state_dict(),
                    'meta_optim_state': maml_model.meta_optim.state_dict()},
                    save_path)

    if args.only_evaluation is False:
        print('Evaluating...')
    
    eval_kwargs = evaluate_kwargs(args)

    print("\nEvaluating for Unseen Domain and Unseen Classes:")
    UU_accuracies, UU_accuracy, UU_variation = evaluate(maml_model, test_set, "Real World", **eval_kwargs)
    print('Accuracy: ' + str(UU_accuracy) + '; Variation: ' + str(UU_variation))

    print("\nEvaluating for Unseen Domain and Seen Classes:")
    US_accuracies, US_accuracy, US_variation = evaluate(maml_model, train_set, "Real World", **eval_kwargs)
    print('Accuracy: ' + str(US_accuracy) + '; Variation: ' + str(US_variation))


    # SU_accuracies_sum = 0
    # SU_accuracy_sum = 0
    # SU_variation_sum = 0
    # print("\nEvaluating for Seen Domain and Unseen Classes:")
    # for eval_domain in ["Art", "Clipart", "Product"]:
    #     print("\tDomain :",eval_domain)        
    #     SU_accuracies, SU_accuracy, SU_variation = evaluate(maml_model, test_set, eval_domain = eval_domain, **eval_kwargs)
    #     print('\tAccuracy: ' + str(SU_accuracy) + '; variation: ' + str(SU_variation))
    #     print("\n")
    #     SU_accuracies_sum += SU_accuracies
    #     SU_accuracy_sum += SU_accuracy
    #     SU_variation_sum += SU_variation
        
    # SU_accuracies_sum = SU_accuracies_sum/3

    if args.only_evaluation is False:
        if not os.path.exists('results_auto/domain_adaptation/gs_maml_supervised/' + fileStart):
            os.makedirs('results_auto/domain_adaptation/gs_maml_supervised/' + fileStart)
        save_path = 'results_auto/domain_adaptation/gs_maml_supervised/' + fileStart + '/' + 'results.npz'
        np.savez(save_path, UU_accuracies=UU_accuracies, US_accuracies=US_accuracies, SU_accuracies=SU_accuracies_sum)

if __name__ == '__main__':
   main()