import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.nn as nn
# from collections import OrderedDict

# '''
# Adopted from arxiv.org/pdf/1906.07774.pdf
# From Appendix B.2:
# > This one is a one hidden layer MLP. Input size is 7 × 7 = 49 and output size is 10. The default number
# of hidden units is 70. We use ReLU activations.
# '''

class Medium_MLP(torch.nn.Module):
    def __init__(self, num_classes=10, num_dim=100):
        super(Medium_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(32*32*3, num_dim)
        self.fc2 = torch.nn.Linear(num_dim, num_dim)
        self.fc3 = torch.nn.Linear(num_dim, num_dim)
        self.fc4 = torch.nn.Linear(num_dim, num_classes)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x

