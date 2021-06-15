"""
Model agnostic meta-learning and evaluation on arbitrary
datasets.
"""

import random
import itertools

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict
from .min_norm_solvers_numpy import MinNormSolver

class MAML:

    def __init__(self, model, device, update_lr, meta_step_size):

        self.device = device
        self.net = model.to(self.device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=meta_step_size)
        self.update_lr = update_lr

    def train_step(self,
                dataset,
                order,
                num_classes,
                num_shots,
                meta_shots,
                inner_iters,
                replacement,
                meta_batch_size):
        """
        Perform a MAML training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """        
        create_graph, retain_graph = True, True
        if order == 1:
            create_graph, retain_graph = False, False

        new_grads = []

        self.meta_optim.zero_grad()

        domain_choices = ["Art", "Clipart", "Product"]
        num_domains = len(domain_choices)

        for i in range(meta_batch_size):

            fast_weight = OrderedDict(self.net.named_parameters())
            
            shuffled = list(dataset)
            random.shuffle(shuffled)
            selected_cl = []
            selected_domain = domain_choices[(i % num_domains)]

            for _, class_obj in enumerate(shuffled[:num_classes]):
                selected_cl.append(class_obj)
            
            train_set=_sample_mini_dataset(selected_cl, num_classes, num_shots, dataset_domain = selected_domain)
            test_set = _sample_mini_dataset(selected_cl, num_classes, meta_shots, dataset_domain = selected_domain)

            inputs, labels = zip(*train_set)
            inputs = (torch.stack(inputs)).to(self.device)
            labels = (torch.tensor(labels)).to(self.device)

            for _ in range(inner_iters):
                
                logits = self.net.functional_forward(inputs, fast_weight)
                fast_loss = F.cross_entropy(logits, labels)
                fast_gradients = torch.autograd.grad(fast_loss, fast_weight.values(), 
                    create_graph=create_graph)

                fast_weight = OrderedDict(
                    (name, param - self.update_lr * grad_param)
                    for ((name, param), grad_param) in zip(fast_weight.items(), fast_gradients))

            inputs, labels = zip(*test_set)
            inputs = (torch.stack(inputs)).to(self.device)
            labels = (torch.tensor(labels)).to(self.device)

            logits = self.net.functional_forward(inputs, fast_weight)
            task_loss = F.cross_entropy(logits, labels)
            task_loss.backward(retain_graph=retain_graph)

            update_gradients_list = []
            for params in self.net.parameters():
                update_gradients_list.append(params.grad.cpu())
            new_grads.append(update_gradients_list)

            self.meta_optim.zero_grad()

        # compute the new gradients.
        # 1. store the sum of new variables in new_grads[0].
        # 2. Find the average parameters.
        # 3. calculate the difference and update the meta learner gradients.
        taskwise_grads = [self.params_to_list(grads) for grads in new_grads]
        mns = MinNormSolver(max_iter=50)

        sol, _ = mns.find_min_norm_element(taskwise_grads)

        mo_grads = []
        for i, grads in enumerate(new_grads):
            if i==0:
                mo_grads = [sol[i]*grad for grad in grads]
            else:
                mo_grads = [sol[i]*grad + mo_grad for grad, mo_grad in zip(grads, mo_grads)]

        counter = 0
        for params in self.net.parameters():
            x = mo_grads[counter]
            params.grad = x.to(self.device)
            counter = counter+1

        # update the meta learner parameters
        self.meta_optim.step()
        self.meta_optim.zero_grad()

    def params_to_list(self, params):
        list_ = []
        for param in params:
            for p in param.view(-1):
                list_.append(p.item())
        return list_

    def evaluate(self,
                dataset,
                num_classes,
                num_shots,
                inner_iters,
                replacement,
                eval_domain = "Real World"):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """

        old_state = deepcopy(self.net.state_dict())
        fast_weight = OrderedDict(self.net.named_parameters())

        shuffled = list(dataset)
        random.shuffle(shuffled)
        selected_cl = []

        for _, class_obj in enumerate(shuffled[:num_classes]):
            selected_cl.append(class_obj)
        
        train_set=_sample_mini_dataset(selected_cl, num_classes, num_shots, dataset_domain = eval_domain)
        test_set = _sample_mini_dataset(selected_cl, num_classes, 1, dataset_domain = eval_domain)        
        
        inputs, labels = zip(*train_set)
        inputs = (torch.stack(inputs)).to(self.device)
        labels = (torch.tensor(labels)).to(self.device)

        for _ in range(inner_iters):

            logits = self.net.functional_forward(inputs, fast_weight)
            fast_loss = F.cross_entropy(logits, labels)
            fast_gradients = torch.autograd.grad(fast_loss, fast_weight.values())

            fast_weight = OrderedDict(
                (name, param - self.update_lr * grad_param)
                for ((name, param), grad_param) in zip(fast_weight.items(), fast_gradients))

        inputs, labels = zip(*test_set)
        inputs = (torch.stack(inputs)).to(self.device)
        labels = (torch.tensor(labels)).to(self.device)

        logits = self.net.functional_forward(inputs, fast_weight)
        test_preds = (F.softmax(logits, dim=1)).argmax(dim=1)
       
        num_correct = torch.eq(test_preds, labels).sum()

        self.net.load_state_dict(old_state)

        return num_correct.item()

            
def _sample_mini_dataset(dataset, num_classes, num_shots, dataset_domain):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    for class_idx, class_obj in enumerate(dataset):
        for sample in class_obj.sample(num_shots, dataset_domain):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set