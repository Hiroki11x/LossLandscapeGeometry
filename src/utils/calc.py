import torch


def get_norm_dot_prod(weights, grads, x_star):
    dist = weights - x_star
    norm_dist = torch.norm(dist)
    norm_grad = torch.norm(grads)
    dot_prod = torch.dot(dist, grads)
    return norm_dist.item(), norm_grad.item(), dot_prod.item()

def get_norm_dot_prod_with_wd(weights, grads, x_star, weight_decay):
    dist = weights - x_star
    grads_wd = grads + weight_decay * weights
    norm_grad_wd = torch.norm(grads_wd)
    dot_prod_wd = torch.dot(dist, grads_wd)
    return norm_grad_wd.item(), dot_prod_wd.item()

def get_weights_and_grads(model):
    params = list(model.parameters())
    weights = []
    grads = []
    for param_group in params:
        weights.append(param_group.flatten().detach())
        grads.append(param_group.grad.flatten().detach())
    weights = torch.cat(weights)
    grads = torch.cat(grads)
    return weights, grads

def get_grads(model):
    params = list(model.parameters())
    grads = []
    for param_group in params:
        grads.append(param_group.grad.flatten().detach())
    grads = torch.cat(grads)
    return grads

def get_weights(model):
    params = list(model.parameters())
    weights = []
    for param_group in params:
        weights.append(param_group.flatten().detach())
    weights = torch.cat(weights)
    return weights