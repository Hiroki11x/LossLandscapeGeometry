# coding: utf-8
import math
import torch

def linear_warmup_cosine_decay(warmup_steps, total_steps):

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn

def multi_step_decay(optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, [25], gamma=0.1) 
