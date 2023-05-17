import os
import torch
import random
import numpy as np
import sys
import torchvision
import numpy as np
import PIL
import sys
import os

from torchsummary import summary

        
def update_dataroot(dataset_name, data_root):
    if dataset_name == 'imagenet':
        data_root = get_local_scratch_path()
    return data_root

def get_local_scratch_path():
    path = f"{os.environ['SLURM_TMPDIR']}"
    return path

def print_libs_version():
    print("\nComputational Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_model_summary(model, dataset):
    print("\nModel Arch Summary:")
    if dataset == 'cifar10':
        summary(model, (3, 32, 32))
    elif dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        return