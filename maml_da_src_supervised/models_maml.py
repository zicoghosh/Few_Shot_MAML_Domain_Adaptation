import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The few-shot classifier model as used in Reptile.
Repeat of block for 4 times:
    conv with 32 channels, a 3*3 kernel, and a stride of 1
    batchnorm 
    maxpool with 2*2 kernel and a stride of 2
    relu activation

a linear layer from flattened output of size 32*5*5 to number of ways.
'''

class MiniImageNetModel(nn.Module):

    def __init__(self, k_way):

        super(MiniImageNetModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch1 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch2 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch3 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch4 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.lin1 = nn.Linear(32*5*5, k_way)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.batch1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.batch2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.batch3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.batch4(self.conv4(x)), 2))
        x = x.view(-1, 32*5*5)
        x = self.lin1(x)
        return x

    def functional_forward(self, x, weight_dict):

        x = F.conv2d(x, weight_dict['conv1.weight'], weight_dict['conv1.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch1.weight'], bias=weight_dict['batch1.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv2.weight'], weight_dict['conv2.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch2.weight'], bias=weight_dict['batch2.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv3.weight'], weight_dict['conv3.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch3.weight'], bias=weight_dict['batch3.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv4.weight'], weight_dict['conv4.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch4.weight'], bias=weight_dict['batch4.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = x.view(-1, 32*5*5)
        
        x = F.linear(x, weight=weight_dict['lin1.weight'], bias=weight_dict['lin1.bias'])

        return x


class OmniglotModel(nn.Module):

    def __init__(self, k_way):

        super(OmniglotModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.batch1 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch2 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch3 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch4 = nn.BatchNorm2d(64, track_running_stats=False)

        self.lin1 = nn.Linear(64, k_way)

    def forward(self, x):
        
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.batch1(self.conv1(x)))

        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.batch2(self.conv2(x)))

        x = F.relu(F.max_pool2d(self.batch3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.batch4(self.conv4(x)), 2))
        
        x = x.view(-1, 64)
        x = self.lin1(x)
        return x

    def functional_forward(self, x, weight_dict):

        x = F.pad(x, (0, 1, 0, 1))
        x = F.conv2d(x, weight_dict['conv1.weight'], weight_dict['conv1.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch1.weight'], bias=weight_dict['batch1.bias'], training=True)
        x = F.relu(x)

        x = F.pad(x, (0, 1, 0, 1))
        x = F.conv2d(x, weight_dict['conv2.weight'], weight_dict['conv2.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch2.weight'], bias=weight_dict['batch2.bias'], training=True)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv3.weight'], weight_dict['conv3.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch3.weight'], bias=weight_dict['batch3.bias'], training=True)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv4.weight'], weight_dict['conv4.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch4.weight'], bias=weight_dict['batch4.bias'], training=True)
        x = F.relu(x)

        x = x.view(-1, 64)

        x = F.linear(x, weight=weight_dict['lin1.weight'], bias=weight_dict['lin1.bias'])

        return x

class Omniglot4SquareModel(nn.Module):

    def __init__(self, k_way):

        super(Omniglot4SquareModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.batch1 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch2 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch3 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch4 = nn.BatchNorm2d(64, track_running_stats=False)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.batch5 = nn.BatchNorm2d(64, track_running_stats=False)

        self.lin1 = nn.Linear(64, k_way)

    def forward(self, x):

        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.batch1(self.conv1(x)))
        
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.batch2(self.conv2(x)))

        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.batch3(self.conv3(x)))

        x = F.relu(F.max_pool2d(self.batch4(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.batch5(self.conv5(x)), 2))
        
        x = x.view(-1, 64)
        x = self.lin1(x)
        return x

    def functional_forward(self, x, weight_dict):

        x = F.pad(x, (0, 1, 0, 1))
        x = F.conv2d(x, weight_dict['conv1.weight'], weight_dict['conv1.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch1.weight'], bias=weight_dict['batch1.bias'], training=True)
        x = F.relu(x)

        x = F.pad(x, (0, 1, 0, 1))
        x = F.conv2d(x, weight_dict['conv2.weight'], weight_dict['conv2.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch2.weight'], bias=weight_dict['batch2.bias'], training=True)
        x = F.relu(x)

        x = F.pad(x, (0, 1, 0, 1))
        x = F.conv2d(x, weight_dict['conv3.weight'], weight_dict['conv3.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch3.weight'], bias=weight_dict['batch3.bias'], training=True)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv4.weight'], weight_dict['conv4.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch4.weight'], bias=weight_dict['batch4.bias'], training=True)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv5.weight'], weight_dict['conv5.bias'], stride=2)
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch5.weight'], bias=weight_dict['batch5.bias'], training=True)
        x = F.relu(x)

        x = x.view(-1, 64)

        x = F.linear(x, weight=weight_dict['lin1.weight'], bias=weight_dict['lin1.bias'])

        return x
