import torch
from torch import nn
from torchvision import models

class Sender(nn.Module):
    def __init__(self, n_hidden) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(3,1,3)
        self.resnet = models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, n_hidden)
        self.relu = nn.ReLU()

    def forward(self, x, _aux_input):
        resnet = self.resnet(x)
        flattened = torch.flatten(self.relu(resnet), start_dim=1)

        return self.linear(flattened)

class Receiver(nn.Module):
    def __init__(self, n_hidden, n_labels) -> None:
        super().__init__()
        self.linear = nn.Linear(n_hidden, n_labels)

    def forward(self, x, _input, _aux_input):
        return self.linear(x)

