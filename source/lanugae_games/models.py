import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ReferentialGameSender(nn.Module):
    def __init__(self, hidden_size, embedding_dimension=256) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # out 2048 * 7 * 7
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # no fine-tuning
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()

        self.adaptive_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_image_1 = nn.Linear(2048, embedding_dimension)
        self.linear_image_2 = nn.Linear(2048, embedding_dimension)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(embedding_dimension * 2, hidden_size)

    def forward(self, x, _aux_input):
        image_1 = self.resnet(x[:, 0])
        pooled_1 = self.adaptive_pool_1(image_1).squeeze()
        linear_1 = self.sigmoid(self.linear_image_1(pooled_1))

        image_2 = self.resnet(x[:, 1])
        pooled_2 = self.adaptive_pool_1(image_2).squeeze()
        linear_2 = self.sigmoid(self.linear_image_1(pooled_2))

        concatenated = torch.cat((linear_1, linear_2), dim=1)

        return self.fc(concatenated)


class ReferentialGameReceiver(nn.Module):
    def __init__(self, hidden_size, embedding_dimension=256) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # out 2048 * 7 * 7
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # no fine-tuning
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()

        self.adaptive_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_image_1 = nn.Linear(2048, embedding_dimension)
        self.linear_image_2 = nn.Linear(2048, embedding_dimension)  #
        self.linear_message = nn.Linear(hidden_size, embedding_dimension)

    def forward(self, x, _input, _aux_input):
        image_1 = self.resnet(_input[:, 0])
        pooled_1 = self.adaptive_pool_1(image_1).squeeze()
        linear_1 = self.linear_image_1(pooled_1)

        image_2 = self.resnet(_input[:, 1])
        pooled_2 = self.adaptive_pool_1(image_2).squeeze()
        linear_2 = self.linear_image_1(pooled_2)

        message = self.linear_message(x)
        dot_product_1 = torch.sum(message * linear_1, dim=1)
        dot_product_2 = torch.sum(message * linear_2, dim=1)

        output = torch.stack((dot_product_1, dot_product_2), dim=1)

        return output
