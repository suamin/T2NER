# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function


class GradientReverseFunction(Function):
    
    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    
    def __init__(self, coeff=1.):
        super().__init__()
        self.coeff = coeff
    
    def forward(self, *input):
        return GradientReverseFunction.apply(*input, self.coeff)


class WarmStartGradientReverseLayer(nn.Module):
    
    def __init__(
        self,
        alpha=1.0,
        lo=0.0,
        hi=1.,
        max_iters=1000.,
        auto_step=False
    ):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
    
    def forward(self, input):
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)
    
    def step(self):
        self.iter_num += 1
