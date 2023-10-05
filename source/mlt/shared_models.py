from abc import ABC

from mlt.feature_extractors import FeatureExtractor
from torch import nn


class ImageEncoder(ABC, nn.Module):
    ...


class ClevrImageEncoder(ImageEncoder):
    def __init__(
        self,
        encoder_out_dim,
        feature_extractor: FeatureExtractor,
    ) -> None:
        super().__init__()
        self.process_image = nn.Sequential(
            feature_extractor,
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(encoder_out_dim))

    def forward(self, image):
        processed_image = self.process_image(image)

        # flattened = torch.flatten(processed_image, start_dim=2).permute(0, 2, 1)
        # reduced = self.mean_reduction(flattened.mean(dim=1))
        reduced = self.reduction(processed_image)

        return reduced


class BoundingBoxImageEncoder(ImageEncoder):
    def __init__(
        self,
        embedding_dimension,
    ) -> None:
        super().__init__()
        self.process_image = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(embedding_dimension, bias=False)
        )

    def forward(self, bounding_box):
        return self.process_image(bounding_box)


class MaskedImageEncoder(ImageEncoder):
    def __init__(
        self,
        encoder_out_dim,
    ) -> None:
        super().__init__()

        self.process_image = nn.Sequential(nn.Flatten(), nn.LazyLinear(encoder_out_dim))

    def forward(self, masked_image):
        return self.process_image(masked_image)


class CoordinateClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LazyLinear(1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LazyLinear(2),
        )

    def forward(self, data):
        return self.classifier(data)
