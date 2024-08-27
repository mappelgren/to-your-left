from abc import ABC, abstractmethod

from torch import nn

from feature_extractors import FeatureExtractor


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

