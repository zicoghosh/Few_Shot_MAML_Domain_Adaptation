"""
Model agnostic meta-learning and evaluation on arbitrary
datasets.
"""

import random
import itertools
import ntpath

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict

import torchvision.utils as vutils

from maml_da_src_supervised.smoothing import LabelSmoothingLoss, LabelSmoothing, SoftTargetCrossEntropy

class MAML:

    def __init__(self, model, device, update_lr, meta_step_size, label_smoothing_enable):

        self.device = device
        self.net = model.to(self.device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=meta_step_size)
        self.update_lr = update_lr
        self.label_smoothing_enable = label_smoothing_enable

        if label_smoothing_enable:
            self.loss_function = SoftTargetCrossEntropy()
            # self.loss_function = LabelSmoothingLoss(smoothing = 0.05)
            self.label_smoother = LabelSmoothing(smoothing = 0.05)



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
            
            # choice_list = list(dataset)
            # random.shuffle(choice_list)
            selected_cl = select_classes(dataset, num_classes)
            selected_domain = domain_choices[(i % num_domains)]


            train_set, test_set = _split_train_test(
                _sample_mini_dataset(selected_cl, num_classes, num_shots+meta_shots, dataset_domain = selected_domain), test_shots=meta_shots)

            # train_set=_sample_mini_dataset(selected_cl, num_classes, num_shots, dataset_domain = selected_domain)
            # test_set = _sample_mini_dataset(selected_cl, num_classes, meta_shots, dataset_domain = selected_domain)

            inputs, labels = zip(*train_set)
            inputs = (torch.stack(inputs)).to(self.device)
            labels = (torch.tensor(labels)).to(self.device)


            if self.label_smoothing_enable:
                labels = self.label_smoother.smooth_target(labels, num_classes)

            for _ in range(inner_iters):
                
                logits = self.net.functional_forward(inputs, fast_weight)
                if self.label_smoothing_enable:
                    fast_loss = self.loss_function.forward(logits, labels)
                else:
                    fast_loss = F.cross_entropy(logits, labels)
                fast_gradients = torch.autograd.grad(fast_loss, fast_weight.values(), 
                    create_graph=create_graph)

                fast_weight = OrderedDict(
                    (name, param - self.update_lr * grad_param)
                    for ((name, param), grad_param) in zip(fast_weight.items(), fast_gradients))

            inputs, labels = zip(*test_set)
            inputs = (torch.stack(inputs)).to(self.device)
            labels = (torch.tensor(labels)).to(self.device)

            if self.label_smoothing_enable:
                labels = self.label_smoother.smooth_target(labels, num_classes)


            logits = self.net.functional_forward(inputs, fast_weight)
            if self.label_smoothing_enable:
                task_loss = self.loss_function.forward(logits, labels)
            else:
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

        for i in range(1, len(new_grads)):
            for params1, params2 in zip(new_grads[0], new_grads[i]):
                params1 = params1.add_(params2)

        cumulative_grad = [wt / meta_batch_size for wt in new_grads[0]]

        counter = 0
        for params in self.net.parameters():
            x = cumulative_grad[counter]
            params.grad = x.to(self.device)
            counter = counter+1

        # update the meta learner parameters
        self.meta_optim.step()
        self.meta_optim.zero_grad()
        

    def evaluate(self,
                dataset,
                num_classes,
                num_shots,
                inner_iters,
                replacement,
                eval_domain):
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

        # shuffled = list(dataset)
        # random.shuffle(shuffled)
        selected_cl = select_classes(dataset, num_classes)

        train_set, test_set = _split_train_test(
            _sample_mini_dataset(selected_cl, num_classes, num_shots + 1, dataset_domain = eval_domain))
   
        # train_set=_sample_mini_dataset(selected_cl, num_classes, num_shots, dataset_domain = eval_domain)
        # test_set = _sample_mini_dataset(selected_cl, num_classes, 1, dataset_domain = eval_domain)

        inputs, labels = zip(*train_set)
        inputs = (torch.stack(inputs)).to(self.device)
        labels = (torch.tensor(labels)).to(self.device)

        if self.label_smoothing_enable:
            labels = self.label_smoother.smooth_target(labels, num_classes)

        
        for _ in range(inner_iters):

            logits = self.net.functional_forward(inputs, fast_weight)
            if self.label_smoothing_enable:
                fast_loss = self.loss_function.forward(logits, labels)
            else:
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


def select_classes(dataset, num_classes):

    selected_cl_names = []
    selected_cl = []

    count = num_classes
    while(count > 0):
        class_obj = random.choice(dataset)
        cl_name = getClassName(class_obj)
        if(cl_name not in selected_cl_names):
            selected_cl.append(class_obj)
            selected_cl_names.append(cl_name)
            count = count - 1
    
    return selected_cl

    

def _sample_mini_dataset(dataset, num_classes, num_shots, dataset_domain):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    for class_idx, class_obj in enumerate(dataset):
        domain = dataset_domain
        for sample in class_obj.sample(num_shots, domain):
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

def getClassName(class_obj):
    dir_path = class_obj.dir_path
    return ntpath.basename(dir_path)