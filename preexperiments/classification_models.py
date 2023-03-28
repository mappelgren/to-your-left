import torch
from torch import nn
from torch.nn import Module
from torchvision.models import ResNet50_Weights, resnet50


class ResnetFeatureClassifier(Module):
    def __init__(self) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet.eval()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(20480, 4096),
            # nn.ReLU(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )   

    def forward(self, data):
        data = data.permute(1, 0, 2, 3, 4)

        after_resnet = []
        for bounding_box in data:
            resnet = self.resnet(bounding_box)
            pooled = self.adaptive_pool(resnet)
            after_resnet.append(pooled)

        stacked = torch.stack(after_resnet)
        stacked = stacked.permute(1, 0, 2, 3, 4)

        classified = self.classifier(torch.flatten(stacked, start_dim=1))

        return classified
