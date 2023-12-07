from abc import ABC, abstractmethod

from mlt.feature_extractors import FeatureExtractor
from torch import nn


class ImageEncoder(ABC, nn.Module):
    @abstractmethod
    def __repr__(self):
        ...


class ClevrImageEncoder(ImageEncoder):
    def __init__(
        self, feature_extractor: FeatureExtractor, max_pool: bool = True
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.max_pool = max_pool

        self.process_image = nn.Sequential(
            feature_extractor,
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        if max_pool:
            self.process_image.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, image):
        return self.process_image(image)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.feature_extractor=},{self.max_pool=})"


class BoundingBoxImageEncoder(ImageEncoder):
    def __init__(
        self,
        image_embedding_dimension,
    ) -> None:
        super().__init__()

        self.image_embedding_dimension = image_embedding_dimension

        self.process_image = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension, bias=False)
        )

    def forward(self, bounding_box):
        return self.process_image(bounding_box)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.image_embedding_dimension=})"


class CoordinateClassifier(nn.Module):
    def __init__(self, coordinate_classifier_dimension) -> None:
        super().__init__()
        self.coordinate_classifier_dimension = coordinate_classifier_dimension

        self.classifier = nn.Sequential(
            nn.LazyLinear(coordinate_classifier_dimension),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LazyLinear(coordinate_classifier_dimension),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LazyLinear(2),
        )

    def forward(self, data):
        return self.classifier(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.coordinate_classifier_dimension=})"


class MaskPredictor(nn.Module):
    def __init__(self, mask_predictor_dimension) -> None:
        super().__init__()
        self.mask_predictor_dimension = mask_predictor_dimension

        self.classifier = nn.Sequential(
            nn.LazyLinear(mask_predictor_dimension),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LazyLinear(mask_predictor_dimension),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LazyLinear(50176),
            nn.Softmax(),
        )

    def forward(self, data):
        return self.classifier(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.mask_predictor_dimension=})"
