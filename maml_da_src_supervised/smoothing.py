# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):

        logprobs = F.log_softmax(x, dim=-1)
        
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        smooth_loss = -logprobs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


class LabelSmoothing():
    
    def __init__(self, smoothing = 0.0):
        self.smoothing = smoothing

    def one_hot(self, x, num_classes, on_value=1., off_value=0., device='cuda'):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)
    
    def smooth_target(self, target, num_classes, lam=1., device='cuda'):
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value        
        y1 = self.one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
        return y1