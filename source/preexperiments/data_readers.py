import itertools
import json
import os
import random
from dataclasses import dataclass
from enum import Enum

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights


class Shape(Enum):
    CUBE = 0
    SPHERE = 1
    CYLINDER = 2

    @staticmethod
    def names():
        return list(map(lambda s: s.name, Shape))


class Color(Enum):
    GRAY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    BROWN = 4
    PURPLE = 5
    CYAN = 6
    YELLOW = 7

    @staticmethod
    def names():
        return list(map(lambda c: c.name, Color))


class Size(Enum):
    SMALL = 0
    LARGE = 1

    @staticmethod
    def names():
        return list(map(lambda s: s.name, Size))


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
        # x, y = self._recalculate_coordinates(image_size, (x, y))

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


class ImageMasker:
    def get_masked_image(self, image, scene, target_object):
        masked_image = image.copy()
        MASK_SIZE = masked_image.size[0] / 5
        x_center, y_center, _ = scene["objects"][target_object]["pixel_coords"]
        pixels = masked_image.load()

        for i, j in itertools.product(
            range(masked_image.size[0]), range(masked_image.size[1])
        ):
            if (
                i < x_center - MASK_SIZE
                or i > x_center + MASK_SIZE
                or j < y_center - MASK_SIZE
                or j > y_center + MASK_SIZE
            ):
                pixels[i, j] = (0, 0, 0)

        return masked_image


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
    masked_image: torch.Tensor = torch.tensor(0)


class CoordinatePredictorDataset(Dataset):
    """
    Input:
     - image
     - attributes (optional)
     - center coordinates of all objects (optional)
     - masked image

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
        mask_image=False,
    ) -> None:
        super().__init__()

        # preprocess = ResNet50_Weights.DEFAULT.transforms()
        preprocess = transforms.Compose([transforms.PILToTensor()])

        coordinate_encoder = CoordinateEncoder(preprocess)
        attribute_encoder = AttributeEncoder()
        image_masker = ImageMasker()

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

            if mask_image:
                masked_image = image_masker.get_masked_image(
                    image, scene, target_object
                )
                sample.masked_image = preprocess(masked_image)

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
                sample.masked_image,
            ),
            sample.target_pixels,
            sample.image_id,
        )

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class CaptionGeneratorSample:
    image_id: str
    image: torch.Tensor

    # target
    caption: torch.Tensor

    # additional attributes
    masked_image: torch.Tensor = torch.tensor(0)
    non_target_captions: torch.Tensor = torch.tensor(0)


class CaptionGeneratorDataset(Dataset):
    """
    Input:
     - image

    Ouput:
     - caption in form of (size, color, shape) e.g. large green sphere
    """

    SOS_TOKEN = "<sos>"

    def __init__(
        self,
        scenes_json_dir,
        image_path,
        max_number_samples,
        mask_image=False,
    ) -> None:
        super().__init__()

        preprocess = ResNet50_Weights.DEFAULT.transforms()
        image_masker = ImageMasker()

        # list instead of set, to make indices deterministic
        vocab = [
            self.SOS_TOKEN,
            *[word.lower() for word in [*Size.names(), *Color.names(), *Shape.names()]],
        ]
        self.vocab = {word: index for index, word in enumerate(list(vocab))}

        self.samples: list[CaptionGeneratorSample] = []

        scenes = os.listdir(scenes_json_dir)
        selected_scenes = random.sample(scenes, max_number_samples)

        for scene_file in selected_scenes:
            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                scene = json.load(f)

            image = Image.open(image_path + scene["image_filename"]).convert("RGB")

            target_object = scene["groups"]["target"][0]
            sos = self.get_encoded_word("<sos>")

            captions = []
            for obj in scene["objects"]:
                size = self.get_encoded_word(obj["size"])
                color = self.get_encoded_word(obj["color"])
                shape = self.get_encoded_word(obj["shape"])
                captions.append(torch.tensor([sos, size, color, shape]))

            target_caption = captions.pop(target_object)
            captions.extend([torch.zeros_like(captions[0])] * (10 - len(captions)))

            sample = CaptionGeneratorSample(
                image_id=scene_file.removesuffix(".json"),
                image=preprocess(image),
                caption=target_caption,
                non_target_captions=torch.stack(captions),
            )

            if mask_image:
                masked_image = image_masker.get_masked_image(
                    image, scene, target_object
                )
                sample.masked_image = preprocess(masked_image)

            self.samples.append(sample)

    def get_encoded_word(self, word):
        return self.vocab[word]

    def get_decoded_word(self, search_index):
        for word, index in self.vocab.items():
            if index == search_index:
                return word

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            (
                sample.image,
                sample.caption,
                sample.non_target_captions[:, 1:],
                sample.masked_image,
            ),
            sample.caption[1:],
            sample.image_id,
        )

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class ReferentialGameSample:
    image_1_id: str
    image_2_id: str
    image_1: torch.Tensor
    image_2: torch.Tensor

    # target
    target_image: torch.Tensor


class ReferentialGameDataset(Dataset):
    def __init__(self, scenes_json_dir, image_path="", max_number_samples="") -> None:
        super().__init__()
        preprocess = ResNet50_Weights.DEFAULT.transforms()

        self.samples: list[ReferentialGameSample] = []

        scenes = os.listdir(scenes_json_dir)
        print("loading scenes...")
        loaded_scenes = []
        for scene_file in scenes:
            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                loaded_scenes.append(json.load(f))

        print("creating combinations...")
        scene_combinations = list(
            itertools.product(range(len(loaded_scenes)), repeat=2)
        )
        print("shuffling...")
        random.shuffle(scene_combinations)

        print("looping...")

        for scene_1_index, scene_2_index in scene_combinations:
            if len(self.samples) == max_number_samples:
                break

            scene_1 = loaded_scenes[scene_1_index]
            scene_2 = loaded_scenes[scene_2_index]

            target_object_1_index = scene_1["groups"]["target"][0]
            target_object_2_index = scene_2["groups"]["target"][0]

            target_object_1 = scene_1["objects"][target_object_1_index]
            target_object_2 = scene_2["objects"][target_object_2_index]

            # no matching attribute
            if (
                target_object_1["color"] == target_object_2["color"]
                or target_object_1["shape"] == target_object_2["shape"]
                or target_object_1["size"] == target_object_2["size"]
            ):
                continue

            image_1 = Image.open(image_path + scene_1["image_filename"]).convert("RGB")
            image_2 = Image.open(image_path + scene_2["image_filename"]).convert("RGB")
            target_image = random.randint(0, 1)

            self.samples.append(
                ReferentialGameSample(
                    image_1_id=scene_1["image_filename"].removesuffix(".png"),
                    image_2_id=scene_2["image_filename"].removesuffix(".png"),
                    image_1=self._get_bounding_box(
                        image_1, scene_1, target_object_1_index
                    ),
                    image_2=self._get_bounding_box(
                        image_2, scene_2, target_object_2_index
                    ),
                    target_image=torch.tensor(target_image),
                )
            )

    def _get_bounding_box(self, image, scene, target_index):
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        BOUNDING_BOX_SIZE = image.size[0] / 5

        x_center, y_center, _ = scene["objects"][target_index]["pixel_coords"]
        bounding_box = image.crop(
            (
                x_center - BOUNDING_BOX_SIZE / 2,
                y_center - BOUNDING_BOX_SIZE / 2,
                x_center + BOUNDING_BOX_SIZE / 2,
                y_center + BOUNDING_BOX_SIZE / 2,
            )
        )

        return preprocess(bounding_box)

    def __getitem__(self, index):
        sample = self.samples[index]
        sender_input = torch.stack((sample.image_1, sample.image_2))

        receiver_input = [sample.image_2]
        receiver_input.insert(sample.target_image, sample.image_1)
        receiver_input = torch.stack(receiver_input)

        target = sample.target_image
        return (sender_input, target, receiver_input)

    def __len__(self) -> int:
        return len(self.samples)
