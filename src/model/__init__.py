# coding: utf-8
import attr
from .resnet_cifar import ResNet18_1_Cifar, ResNet18_2_Cifar, ResNet18_3_Cifar
from .resnet_imagenet import ResNet18_1, ResNet18_2, ResNet18_3, ResNet50_1, ResNet50_2, ResNet152_05, ResNet152_1
from .mlp import Medium_MLP
from .transformer import TransformerModel

@attr.s
class ModelSetting:
    name = attr.ib()
    num_classes = attr.ib()


def build_model(setting: ModelSetting):
    name = setting.name
    num_classes = setting.num_classes

    # CIFAR10    
    if name == 'medium_mlp':
        model = Medium_MLP(num_classes, num_dim=4096)
        return model

    if name == 'resnet18_1_cifar':
        model = ResNet18_1_Cifar(num_classes)
        return model

    if name == 'resnet18_2_cifar':
        model = ResNet18_2_Cifar(num_classes)
        return model

    if name == 'resnet18_3_cifar':
        model = ResNet18_3_Cifar(num_classes)
        return model

    # ImageNet
    if name == 'resnet18_1':
        model = ResNet18_1(num_classes)
        return model

    if name == 'resnet18_2':
        model = ResNet18_2(num_classes)
        return model

    if name == 'resnet18_3':
        model = ResNet18_3(num_classes)
        return model

    if name == 'resnet50_1':
        model = ResNet50_1(num_classes)
        return model

    if name == 'resnet50_2':
        model = ResNet50_2(num_classes)
        return model

    if name == 'resnet152_05':
        model = ResNet152_05(num_classes)
        return model

    if name == 'resnet152_1':
        model = ResNet152_1(num_classes)
        return model

    raise ValueError(f'The selected model {name} is not supported for this implementation.')


def build_nlp_model(exp_dict, ntokens):
    device = exp_dict['device']

    if exp_dict['model'] == 'Transformer':
        nlp_model = TransformerModel(ntoken=ntokens, 
                                     ninp=exp_dict['emsize'],
                                     nhead=4, 
                                     nhid=exp_dict['nhid'], 
                                     nlayers=12, 
                                     dropout=exp_dict['dropout']).to(device)
    
    return nlp_model