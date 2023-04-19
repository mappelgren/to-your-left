import json
import os
import random
from dataclasses import dataclass
from enum import Enum

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights


class Shape(Enum):
    CUBE = 0
    SPHERE = 1
    CYLINDER = 2


class Color(Enum):
    GRAY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    BROWN = 4
    PURPLE = 5
    CYAN = 6
    YELLOW = 7


class Size(Enum):
    SMALL = 0
    LARGE = 1


class BoundingBoxClassifierDataset(Dataset):
    """
    Input:
     - bounding boxes of all objects

    Ouput:
     - index of target bounding box
    """

    def __init__(self, scenes_json_dir, image_path, max_number_samples) -> None:
        super().__init__()

        self.samples = []

        scenes = os.listdir(scenes_json_dir)
        selected_scenes = random.sample(scenes, max_number_samples)

        for scene_file in selected_scenes:
            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                scene = json.load(f)

            image = Image.open(image_path + scene["image_filename"]).convert("RGB")

            bounding_boxes = self._get_bounding_boxes(image, scene)

            target_object = scene["groups"]["target"][0]
            enumerated = list(enumerate(bounding_boxes))
            random.shuffle(enumerated)

            input_boxes = torch.stack([bounding_box for _, bounding_box in enumerated])
            indices, _ = zip(*enumerated)
            target_index = torch.tensor(indices.index(target_object))

            self.samples.append(
                (input_boxes, target_index, scene_file.removesuffix(".json"))
            )

    def _get_bounding_boxes(self, image, scene):
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        BOUNDING_BOX_SIZE = image.size[0] / 5

        object_bounding_boxes = []
        for obj in scene["objects"]:
            x_center, y_center, _ = obj["pixel_coords"]
            bounding_box = image.crop(
                (
                    x_center - BOUNDING_BOX_SIZE / 2,
                    y_center - BOUNDING_BOX_SIZE / 2,
                    x_center + BOUNDING_BOX_SIZE / 2,
                    y_center + BOUNDING_BOX_SIZE / 2,
                )
            )
            object_bounding_boxes.append(preprocess(bounding_box))

        # magic number 10 (max objects in scene)
        object_bounding_boxes.extend(
            [torch.zeros_like(object_bounding_boxes[0])]
            * (10 - len(object_bounding_boxes))
        )

        return object_bounding_boxes

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class AttributeEncoder:
    def encode(self, scene, object_index):
        color_tensor = self._one_hot_encode(
            Color, scene["objects"][object_index]["color"]
        )
        shape_tensor = self._one_hot_encode(
            Shape, scene["objects"][object_index]["shape"]
        )
        size_tensor = self._one_hot_encode(Size, scene["objects"][object_index]["size"])

        return color_tensor, shape_tensor, size_tensor

    def _one_hot_encode(self, attribute: Enum, value: str):
        tensor = torch.zeros(len(attribute))
        tensor[attribute[value.upper()].value] = 1

        return tensor


class CoordinateEncoder:
    def __init__(self, preprocess) -> None:
        self.preprocess = preprocess

    def get_object_coordinates(self, object_index, scene, image_size):
        x, y, _ = scene["objects"][object_index]["pixel_coords"]
        x, y = self._recalculate_coordinates(image_size, (x, y))

        return x, y

    def get_locations(self, scene, image_size):
        locations = []
        for index, _ in enumerate(scene["objects"]):
            x, y = self.get_object_coordinates(index, scene, image_size)
            locations.append(torch.tensor([x, y]))
        locations.extend([torch.zeros_like(locations[0])] * (10 - len(locations)))
        random.shuffle(locations)

        return locations

    def _recalculate_coordinates(self, image_size, object_pixels):
        old_x, old_y = object_pixels
        image_x, image_y = image_size

        new_image_x = min(image_x, self.preprocess.resize_size[0])
        new_image_y = min(image_y, self.preprocess.resize_size[0])

        new_x = int(old_x * (new_image_x / image_x))
        new_y = int(old_y * (new_image_y / image_y))

        new_x = int(new_x - ((new_image_x - self.preprocess.crop_size[0]) / 2))
        new_y = int(new_y - ((new_image_y - self.preprocess.crop_size[0]) / 2))

        return new_x, new_y


@dataclass
class CoordinatePredictorSample:
    image_id: str
    image: torch.Tensor

    # target
    target_pixels: torch.Tensor

    # addtional (optional) information
    color_tensor: torch.Tensor = torch.tensor(0)
    shape_tensor: torch.Tensor = torch.tensor(0)
    size_tensor: torch.Tensor = torch.tensor(0)
    locations: torch.Tensor = torch.tensor(0)


class CoordinatePredictorDataset(Dataset):
    """
    Input:
     - image
     - attributes (optional)
     - center coordinates of all objects (optional)

    Ouput:
     - x and y coordinate of target object
    """

    def __init__(
        self,
        scenes_json_dir,
        image_path,
        max_number_samples,
        encode_attributes=False,
        encode_locations=False,
    ) -> None:
        super().__init__()

        preprocess = ResNet50_Weights.DEFAULT.transforms()
        coordinate_encoder = CoordinateEncoder(preprocess)
        attribute_encoder = AttributeEncoder()

        self.samples: list[CoordinatePredictorSample] = []

        scenes = os.listdir(scenes_json_dir)
        selected_scenes = random.sample(scenes, max_number_samples)

        for scene_file in selected_scenes:
            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                scene = json.load(f)

            image = Image.open(image_path + scene["image_filename"]).convert("RGB")

            target_object = scene["groups"]["target"][0]
            target_x, target_y = coordinate_encoder.get_object_coordinates(
                target_object, scene, image.size
            )

            sample = CoordinatePredictorSample(
                image_id=scene_file.removesuffix(".json"),
                image=preprocess(image),
                target_pixels=torch.tensor([target_x, target_y]),
            )

            if encode_attributes:
                (
                    sample.color_tensor,
                    sample.shape_tensor,
                    sample.size_tensor,
                ) = attribute_encoder.encode(scene, target_object)

            if encode_locations:
                sample.locations = torch.cat(
                    coordinate_encoder.get_locations(scene, image.size)
                )

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            (
                sample.image,
                sample.color_tensor,
                sample.shape_tensor,
                sample.size_tensor,
                sample.locations,
            ),
            sample.target_pixels,
            sample.image_id,
        )

    def __len__(self) -> int:
        return len(self.samples)
