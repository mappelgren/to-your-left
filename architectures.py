import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet, ResNet50_Weights, resnet50
from torchvision.models.resnet import Bottleneck


class Sender(nn.Module):
    def __init__(self, n_hidden, encoded_image_size=1) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(3,3,3)

        resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        # resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.linear = nn.Linear(20480, n_hidden)
        self.relu = nn.ReLU()

    def forward(self, x, _aux_input):
        # permute from [batch, n_bounding_box, channels, pixel_x, pixel_y]
        #           to [n_bounding_box, batch, channels, pixel_x, pixel_y]
        
        x = x.permute(1, 0, 2, 3, 4)

        after_resnet = []
        for obj in x:
            resnet = self.resnet(obj)
            after_resnet.append(self.adaptive_pool(resnet))

        stacked = torch.stack(after_resnet)

        # permute back
        stacked = stacked.permute(1, 0, 2, 3, 4)
        flattend = stacked.flatten(start_dim=1)
        return self.relu(self.linear(flattend))

class Receiver(nn.Module):
    def __init__(self, n_hidden, n_labels, encoded_image_size=1) -> None:
        super().__init__()
        resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        # resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))        
        
        self.linear_1 = nn.Linear(20480 + n_hidden, n_labels)
        # self.linear_1 = nn.Linear(20480 + n_hidden, 10000)
        # self.linear_2 = nn.Linear(10000, 5000)
        # self.linear_3 = nn.Linear(5000, n_labels)
        self.relu = nn.ReLU()

    def forward(self, x, _input, _aux_input):
        # permute from [batch, n_bounding_box, channels, pixel_x, pixel_y]
        #           to [n_bounding_box, batch, channels, pixel_x, pixel_y]
        _input = _input.permute(1, 0, 2, 3, 4)
        after_resnet = []
        for obj in _input:
            resnet = self.resnet(obj)
            after_resnet.append(self.adaptive_pool(resnet))

        stacked = torch.stack(after_resnet)

        # permute back
        stacked = stacked.permute(1, 0, 2, 3, 4)
        flattend = stacked.flatten(start_dim=1)

        concat = torch.cat((flattend, x), dim=1)

        # linear_1 = self.linear_1(self.relu(concat))
        # linear_2 = self.linear_2(self.relu(linear_1))
        # out = self.linear_3(self.relu(linear_2))
        out = self.linear_1(self.relu(concat))
        return out

