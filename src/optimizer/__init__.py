# coding: utf-8
import attr
import torch.optim as optim

@attr.s
class OptimizerSetting:
    name = attr.ib()
    lr = attr.ib()
    weight_decay = attr.ib()
    model = attr.ib()

    momentum = attr.ib(default=0.9) # sgd, sgd_nesterov
    eps = attr.ib(default=0.001) # adam, rmsprop (term added to the denominator to improve numerical stability )
    beta_1 = attr.ib(default=0.9) #adam
    beta_2 = attr.ib(default=0.999) #adam

def build_optimizer(setting: OptimizerSetting):
    name = setting.name
    model_params = setting.model.parameters()
    model = setting.model

    # Standard Optimizer
    if name == 'vanilla_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        weight_decay=setting.weight_decay)

    elif name == 'momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum,
                        weight_decay=setting.weight_decay)

    elif name == 'nesterov_momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum, 
                        weight_decay=setting.weight_decay, 
                        nesterov=True)

    elif name == 'adam':
        return optim.Adam(params=model_params, 
                        lr=setting.lr, 
                        betas=(setting.beta_1, setting.beta_2), 
                        eps=setting.eps, 
                        weight_decay=setting.weight_decay, 
                        amsgrad=False)

    else:
        raise ValueError(
            'The selected optimizer is not supported for this trainer.')
        