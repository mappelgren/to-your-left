import os
from abc import ABC, abstractmethod

from PIL import Image
from wandb.old.summary import h5py
import torch


class ImageLoader(ABC):
    @abstractmethod
    def get_image(self, image_id):
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

class FeatureImageLoader(ImageLoader):
    def __init__(self, feature_file, image_dir) -> None:
        self.image_dir = image_dir
        self.feature_file = feature_file

        with h5py.File(feature_file, "r") as f:
            feature_data_set = f["features"]
            self.image_size = feature_data_set.attrs["image_size"]

    def get_image(self, image_id):
        image_index = int(image_id[-6:])

        with h5py.File(self.feature_file, "r") as f:
            feature_data_set = f["features"]
            features = feature_data_set[image_index]

        image = Image.open(os.path.join(self.image_dir, image_id + ".png")).convert(
            "RGB"
        )

        return image, torch.from_numpy(features), self.image_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.feature_file=}, {self.image_dir=})"
