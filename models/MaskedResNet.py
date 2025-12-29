import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


activation_dict = {'relu': F.relu,
                   'sigmoid': F.sigmoid}


class Bern(torch.autograd.Function):
    """
    Custom Bernouli function that supports gradients.
    The original Pytorch implementation of Bernouli function,
    does not support gradients.

    First-Order gradient of bernouli function with prbabilty p, is p.

    Inputs: Tensor of arbitrary shapes with bounded values in [0,1] interval
    Outputs: Randomly generated Tensor of only {0,1}, given Inputs as distributions.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        pvals = ctx.saved_tensors
        return pvals[0] * grad_output


class MaskedLinear(nn.Linear):
    """
        Implementation of masked linear layer, with training strategy in
        https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
    """

    def __init__(self, in_features, out_features, init='ME_init', device=None, **kwargs):
        super(MaskedLinear, self).__init__(in_features, out_features, device=device, **kwargs)
        self.device = device
        arr_weights = None
        self.device = device
        self.init = init
        self.c = np.e * np.sqrt(1 / in_features)
        # Different weight initialization distributions
        if init == 'ME_init':
            arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
        elif init == 'ME_init_sym':
            arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
            arr_weights = np.triu(arr_weights, k=1) + np.tril(arr_weights)
        elif init == 'uniform':
            arr_weights = np.random.uniform(-self.c, self.c, size=(out_features, in_features)) * np.sqrt(3)
        elif init == 'k_normal':
            arr_weights = np.random.normal(0, self.c, size=(out_features, in_features))

        self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, device=self.device,
                                                dtype=torch.float))

        arr_bias = np.random.choice([-self.c, self.c], size=out_features)
        self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=False, device=self.device,
                                                dtype=torch.float))

        # Weights of Mask
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True, device=self.device))
        self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True, device=self.device))

    def forward(self, x, ths=None):
        if ths is None:
            # Generate probability of bernouli distributions
            s_m = torch.sigmoid(self.mask)
            s_b_m = torch.sigmoid(self.bias_mask)
            g_m = Bern.apply(s_m)
            g_b_m = Bern.apply(s_b_m)
        else:
            nd_w_mask = torch.sigmoid(self.mask)
            nd_b_mask = torch.sigmoid(self.bias_mask)
            g_m = torch.where(nd_w_mask > ths, 1, 0)
            g_b_m = torch.where(nd_b_mask > ths, 1, 0)

        # Compute element-wise product with mask
        effective_weight = self.weight * g_m
        effective_bias = self.bias * g_b_m

        # Apply the effective weight on the input data
        lin = F.linear(x, effective_weight.to(self.device), effective_bias.to(self.device))
        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.weight.sum(), torch.abs(self.weight).sum(),
                                                                    self.mask.sum() / prod)

class MaskedConv2d(nn.Conv2d):
    """
        Implementation of masked convolutional layer, with training strategy in
        https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, init='ME_init', device=None, **kwargs):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, device=device, **kwargs)
        self.device = device
        arr_weights = None
        self.init = init
        self.c = np.e * np.sqrt(1/(kernel_size**2 * in_channels))

        if init == 'ME_init':
            arr_weights = np.random.choice([-self.c, self.c],
                                           size=(out_channels, in_channels, kernel_size, kernel_size))
        elif init == 'uniform':
            arr_weights = np.random.uniform(-self.c, self.c,
                                            size=(out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(3)
        elif init == 'k_normal':
            arr_weights = np.random.normal(0, self.c ** 2, size=(out_channels, in_channels, kernel_size, kernel_size))

        self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, device=self.device,
                                                dtype=torch.float))

        arr_bias = np.random.choice([-self.c, self.c], size=out_channels)
        self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=False, device=self.device, dtype=torch.float))

        self.mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True, device=self.device))
        self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True, device=self.device))
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x, ths=None):

        if ths is None:
            # Generate probability of bernouli distributions
            s_m = torch.sigmoid(self.mask)
            s_b_m = torch.sigmoid(self.bias_mask)
            g_m = Bern.apply(s_m)
            g_b_m = Bern.apply(s_b_m)
        else:
            nd_w_mask = torch.sigmoid(self.mask)
            nd_b_mask = torch.sigmoid(self.bias_mask)
            g_m = torch.where(nd_w_mask > ths, 1, 0)
            g_b_m = torch.where(nd_b_mask > ths, 1, 0)

        effective_weight = self.weight * g_m
        effective_bias = self.bias * g_b_m
        # Apply the effective weight on the input data
        lin = self._conv_forward(x, effective_weight.to(self.device), effective_bias.to(self.device))

        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.weight.sum(), torch.abs(self.weight).sum(),
                                                                    self.mask.sum() / prod)

class MaskedBatchNorm2d(nn.BatchNorm2d):
    size = 0
    def forward(self, x):
        return super().forward(x)
    
    
class MaskedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MaskedBasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(planes)

        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MaskedBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MaskedBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MaskedClientResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(MaskedClientResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out
    
    
class MaskedServerResNet(nn.Module):
    def __init__(self, block, num_blocks, n_class=100):
        super(MaskedServerResNet, self).__init__()
        self.in_planes = 128

        self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, n_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.layer3(x)
        out = self.layer4(out)
        # print(out.size())
        # out = F.avg_pool2d(out, 2)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)

        return out
    

# Define the ResNet-18 backbone
def MaskedClientResNet18():
    return MaskedClientResNet(MaskedBasicBlock, [2,2])

def MaskedServerResNet18(n_class):
    return MaskedServerResNet(MaskedBasicBlock, [2,2], n_class=n_class)

