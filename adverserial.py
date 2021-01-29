import torch
from torch import nn
from utils import init_weights
import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def adv_loss(features, ad_net, device='cpu'):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    adv_accuracy = torch.isclose(ad_out.round(), dc_target).sum() / float(ad_out.size(0))
    return nn.BCELoss()(ad_out, dc_target), adv_accuracy


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, max_iter, lr_mult):
        """

        Args:
            in_feature:
            hidden_size:
            max_iter:
            lr_mult: Only used in inv_lr_scheduler in utils.py!!
        """
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = float(max_iter)
        self.lr_mult = lr_mult

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": self.lr_mult}]


