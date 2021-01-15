# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .tagger import Tagger
from .grl import GradientReverseLayer, WarmStartGradientReverseLayer


class SoftmaxClassifier(nn.Module):
    
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)
    
    def forward(self, x, label=None):
        logits = self.linear(x)
        outputs = {"logits": logits}
        if label is not None:
            outputs["loss"] = nn.CrossEntropyLoss()(logits, label)
        return outputs


class BinaryClassifier(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x, label=None):
        logits = self.linear(x).squeeze_(-1)
        outputs = {"logits": logits}
        if label is not None:
            outputs["loss"] = nn.BCEWithLogitsLoss()(logits, label.float())
        return outputs


def get_clf_head(input_dim, num_labels):
    if num_labels == 2:
        return BinaryClassifier(input_dim)
    else:
        return SoftmaxClassifier(input_dim, num_labels)


class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers=1, dropout=0.):
        super().__init__()
        net = list()
        for _ in range(num_layers - 1):
            layers = list()
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            net.extend(layers)
        net.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x).squeeze_()


class WassersteinCritic(MLP):
    
    def __init__(self, input_dim, output_dim=1, num_layers=1, dropout=0.):
        super().__init__(input_dim, output_dim, num_layers, dropout)
    
    def forward(self, x):
        fx = self.net(x)
        outputs = {"logits": fx, "loss": fx.mean()}
        return outputs


class SeqWassersteinCritic(MLP):
    
    def __init__(self, input_dim, output_dim=1, num_layers=1, dropout=0.):
        super().__init__(input_dim, output_dim, num_layers, dropout)
    
    def forward(self, seq, mask=None):
        # seq : batch_size x seq_len x hidden_dim
        # mask : batch_size x seq_len
        fx = self.net(seq)
        if mask is not None:
            fx = fx * mask.unsqueeze(-1)
        fx = fx.sum(-1)
        outputs = {"logits": fx, "loss": fx.mean()}
        return outputs


def get_grl(coeff=-1, **scheduler_kwargs):
    if coeff > 0.:
        grl = GradientReverseLayer(coeff)
    else:
        grl = WarmStartGradientReverseLayer(
            alpha=scheduler_kwargs.get("gamma", 10.),
            lo=scheduler_kwargs.get("lo", 0.),
            hi=scheduler_kwargs.get("hi", 1.),
            max_iters=scheduler_kwargs.get("max_iters", 1000.),
            auto_step=scheduler_kwargs.get("auto_step", True)
        )
    return grl


Classifier = get_clf_head
TokenClassifier = Tagger
GRL = get_grl


class BinaryAdvClassifier(nn.Module):
    
    def __init__(self, input_dim, coeff=-1, **scheduler_kwargs):
        super().__init__()
        self.grl = GRL(coeff, **scheduler_kwargs)
        self.classifier = BinaryClassifier(input_dim)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, xs, xt, apply_grl=True, flip=False):
        x = torch.cat((xs, xt), dim=0)
        if apply_grl:
            x = self.grl(x)
        h = self.classifier(x)["logits"]
        hs, ht = h.chunk(2, dim=0)
        ls, lt = get_dummy_labels(hs.shape, ht.shape, xs.device, flip)
        outputs = {
            "loss": 0.5 * (self.bce(hs, ls.float()) + self.bce(ht, lt.float())),
            "acc": 0.5 * (binary_accuracy(hs, ls) + binary_accuracy(ht, lt))
        }
        return outputs


def get_dummy_labels(shape_s, shape_t, device, flip=False):
    if len(shape_s) > 1:
        shape_s = shape_s[:-1]
    if len(shape_t) > 1:
        shape_t = shape_t[:-1]
    if not flip:
        return torch.ones(shape_s).to(device), torch.zeros(shape_t).to(device)
    else:
        return torch.zeros(shape_s).to(device), torch.ones(shape_t).to(device)


def binary_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        output = nn.Sigmoid()(output)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct
