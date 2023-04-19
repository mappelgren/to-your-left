import torch
from torch import nn
from torch.nn import Module
from torchvision.models import ResNet50_Weights, resnet50


class AbstractResnet(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # out 2048 * 7 * 7
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet.eval()


class BoundingBoxClassifier(AbstractResnet):
    """
    Output:
     - classified bounding box (10 dimensions)

    Input:
     - bounding boxes of objects
    """

    def __init__(self) -> None:
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(20480, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1),
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


class CoordinatePredictor(AbstractResnet):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
    """

    def __init__(self) -> None:
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(100352, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, data):
        image, *_ = data

        resnet = self.resnet(image)
        pooled = self.adaptive_pool(resnet)
        classified = self.classifier(torch.flatten(pooled, start_dim=1))

        return classified


class AttributeCoordinatePredictor(AbstractResnet):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
     - attributes (shape, size, color)
    """

    def __init__(self, number_colors=8, number_shapes=3, number_size=2) -> None:
        super().__init__()
        # out 100_352
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reduction = nn.Linear(100_352, 2048)

        self.predictor = nn.Linear(
            2048 + number_colors + number_shapes + number_size, 2
        )

    def forward(self, data):
        image, color_tensor, shape_tensor, size_tensor, *_ = data
        resnet = self.resnet(image)
        pooled = self.adaptive_pool(resnet)
        reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        concatenated = torch.cat(
            (reduced, color_tensor, shape_tensor, size_tensor), dim=1
        )
        predicted = self.predictor(concatenated)

        return predicted


class AttributeLocationCoordinatePredictor(AbstractResnet):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
     - attributes (shape, size, color)
     - center coordinates of all objects
    """

    def __init__(
        self, number_colors=8, number_shapes=3, number_size=2, number_objects=10
    ) -> None:
        super().__init__()
        # out 100_352
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reduction = nn.Linear(100_352, 2048)

        self.predictor = nn.Linear(
            2048 + number_colors + number_shapes + number_size + (number_objects * 2),
            2,
        )

    def forward(self, data):
        image, color_tensor, shape_tensor, size_tensor, locations = data
        resnet = self.resnet(image)
        pooled = self.adaptive_pool(resnet)
        reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        concatenated = torch.cat(
            (reduced, color_tensor, shape_tensor, size_tensor, locations), dim=1
        )
        predicted = self.predictor(concatenated)

        return predicted
