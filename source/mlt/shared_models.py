from abc import ABC, abstractmethod

from mlt.feature_extractors import FeatureExtractor
from torch import nn


class ImageEncoder(ABC, nn.Module):
    @abstractmethod
    def __repr__(self):
        ...


class ClevrImageEncoder(ImageEncoder):
    def __init__(
        self,
        encoder_out_dim,
        feature_extractor: FeatureExtractor,
    ) -> None:
        super().__init__()
        self.encoder_out_dim = encoder_out_dim
        self.feature_extractor = feature_extractor

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
        return processed_image
        # flattened = torch.flatten(processed_image, start_dim=2).permute(0, 2, 1)
        # reduced = self.mean_reduction(flattened.mean(dim=1))
        reduced = self.reduction(processed_image)

        return reduced

    def __repr__(self):
        return f"{self.__class__.__name__}({self.encoder_out_dim=}, {self.feature_extractor=})"


class BoundingBoxImageEncoder(ImageEncoder):
    def __init__(
        self,
        embedding_dimension,
    ) -> None:
        super().__init__()

        self.embedding_dimensions = embedding_dimension

        self.process_image = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(embedding_dimension, bias=False)
        )

    def forward(self, bounding_box):
        return self.process_image(bounding_box)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.embedding_dimension=})"


class MaskedImageEncoder(ImageEncoder):
    def __init__(
        self,
        encoder_out_dim,
    ) -> None:
        super().__init__()

        self.encoder_out_dim = encoder_out_dim

        self.process_image = nn.Sequential(nn.Flatten(), nn.LazyLinear(encoder_out_dim))

    def forward(self, masked_image):
        return self.process_image(masked_image)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.encoder_out_dim=})"


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

    def __repr__(self):
        return f"{self.__class__.__name__}()"
