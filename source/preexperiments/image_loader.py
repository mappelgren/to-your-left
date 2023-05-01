import os
from abc import ABC, abstractmethod

import h5py
import torch
from PIL import Image


class ImageLoader(ABC):
    @abstractmethod
    def get_image(self, image_id):
        ...


class ClevrImageLoader(ImageLoader):
    def __init__(self, image_dir, preprocess) -> None:
        self.image_dir = image_dir
        self.preprocess = preprocess

    def get_image(self, image_id):
        image = Image.open(os.path.join(self.image_dir, image_id + ".png")).convert(
            "RGB"
        )
        return image, self.preprocess(image), image.size


class FeatureImageLoader(ImageLoader):
    def __init__(self, feature_file) -> None:
        with h5py.File(feature_file, "r") as f:
            feature_data_set = f["features"]
            self.image_size = feature_data_set.attrs["image_size"]
            self.features = list(feature_data_set)

    def get_image(self, image_id):
        image_index = int(image_id[-6:])
        return torch.tensor(0), self.features[image_index], self.image_size
