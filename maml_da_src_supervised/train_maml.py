"""
Training helpers for supervised meta-learning.
"""

import os
import numpy as np

import torch

def train(maml_model,
        train_set,
        val_set,
        test_set,
        model_save_path=None,
        order=2,
        num_classes=5,
        num_shots=1,
        meta_shots=15,
        inner_iters=20,
        replacement=False,
        meta_batch_size=1,
        meta_iters=40000,
        eval_inner_iters=50,
        eval_interval=100,
        num_samples=10000,
        eval_interval_sample=100,
        sample_acc_interval_domain = "Product"):
    
    """
    Train a model on a dataset.
    """
    train_accuracy, test_accuracy = [], []
    if(sample_acc_interval_domain == "RealWorld"):
        sample_acc_interval_domain = "Real World"

    if val_set is not None:
        val_accuracy = []

    for i in range(meta_iters):

        maml_model.train_step(train_set, order=order,
                        num_classes=num_classes, num_shots=num_shots,
                        meta_shots=meta_shots,
                        inner_iters=inner_iters,
                        replacement=replacement, 
                        meta_batch_size=meta_batch_size)
        
        
        if i % eval_interval == 0:
            
            total_correct = 0
            for _ in range(eval_interval_sample):
                total_correct += maml_model.evaluate(train_set,
                                    num_classes=num_classes, num_shots=num_shots,
                                    inner_iters=eval_inner_iters, replacement=replacement, eval_domain = sample_acc_interval_domain)
            
            train_accuracy.append(total_correct / (eval_interval_sample * num_classes))
            
            total_correct = 0
            for _ in range(eval_interval_sample):
                total_correct += maml_model.evaluate(test_set,
                                    num_classes=num_classes, num_shots=num_shots,
                                    inner_iters=eval_inner_iters, replacement=replacement, eval_domain = sample_acc_interval_domain)
            
            test_accuracy.append(total_correct / (eval_interval_sample * num_classes))

            save_path = model_save_path + '/intermediate_' + str(i) + '_model.pt'

            torch.save({'model_state': maml_model.net.state_dict(),
                        'meta_optim_state': maml_model.meta_optim.state_dict()},
                       save_path)

            if val_set is not None:
                total_correct = 0
                for _ in range(eval_interval_sample):
                    total_correct += maml_model.evaluate(val_set,
                                    num_classes=num_classes, num_shots=num_shots,
                                    inner_iters=eval_inner_iters, replacement=replacement, eval_domain = sample_acc_interval_domain)
                val_accuracy.append(total_correct / (eval_interval_sample * num_classes))
            
                print('batch %d: train=%f val=%f test=%f' % (i, 
                    train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))
            else:
                print('batch %d: train=%f test=%f' % (i,
                    train_accuracy[-1], test_accuracy[-1]))

    res_save_path = model_save_path + '/' + 'intermediate_accuracies.npz'

    if val_set is not None:
        np.savez(res_save_path, train_accuracy=np.array(train_accuracy),
            val_accuracy=np.array(val_accuracy), test_accuracy=np.array(test_accuracy))
    else:
        np.savez(res_save_path, train_accuracy=np.array(train_accuracy),
            test_accuracy=np.array(test_accuracy))