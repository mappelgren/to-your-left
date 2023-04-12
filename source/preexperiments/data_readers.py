import json
import random
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

class ClassifierDataset(Dataset):
    def __init__(self, scenes_json_file, image_path, max_number_samples) -> None:
        super().__init__()

        self.samples = []

        with open(scenes_json_file, 'r', encoding='utf-8') as f:
            scenes_metadata = json.load(f)
        for index, scene in enumerate(scenes_metadata['scenes']):
            if index == max_number_samples:
                break

            image = Image.open(image_path + scene['image_filename']).convert('RGB')

            bounding_boxes = self._get_bounding_boxes(image, scene)

            target_object = scene['groups']['target'][0]
            enumerated = list(enumerate(bounding_boxes))
            random.shuffle(enumerated)

            input_boxes = torch.stack([bounding_box for _, bounding_box in enumerated])
            indices, _ = zip(*enumerated)
            target_index = torch.tensor(indices.index(target_object))
            
            self.samples.append((input_boxes, target_index))
            


    def _get_bounding_boxes(self, image, scene):
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        BOUNDING_BOX_SIZE = image.size[0] / 5

        object_bounding_boxes = []
        for obj in scene['objects']:
            x_center, y_center, _ = obj['pixel_coords']
            bounding_box = image.crop((
                x_center - BOUNDING_BOX_SIZE/2,
                y_center - BOUNDING_BOX_SIZE/2,
                x_center + BOUNDING_BOX_SIZE/2,
                y_center + BOUNDING_BOX_SIZE/2
                ))
            object_bounding_boxes.append(preprocess(bounding_box))

        object_bounding_boxes.extend([torch.zeros_like(object_bounding_boxes[0])] * (10 - len(object_bounding_boxes)))

        return object_bounding_boxes

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

class AttentionDataset(Dataset):
    def __init__(self, scenes_json_file, image_path, max_number_samples) -> None:
        super().__init__()

        preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.samples = []

        with open(scenes_json_file, 'r', encoding='utf-8') as f:
            scenes_metadata = json.load(f)
        for index, scene in enumerate(scenes_metadata['scenes']):
            if index == max_number_samples:
                break

            image = Image.open(image_path + scene['image_filename']).convert('RGB')

            target_object = scene['groups']['target'][0]
            target_x, target_y, _ = scene['objects'][target_object]['pixel_coords']

            target_x, target_y = self._recalculate_target_pixels(image.size, (target_x, target_y), preprocess)

            self.samples.append((preprocess(image), torch.tensor([target_x, target_y])))

    def _recalculate_target_pixels(self, image_size, target_pixels, preprocess):
        target_x, target_y = target_pixels
        image_x, image_y = image_size

        new_image_x = min(image_x, preprocess.resize_size[0])
        new_image_y = min(image_y, preprocess.resize_size[0])

        new_x = int(target_x * (new_image_x / image_x))
        new_y = int(target_y * (new_image_y / image_y))

        new_x = int(new_x - ((new_image_x - preprocess.crop_size[0]) / 2))
        new_y = int(new_y - ((new_image_y - preprocess.crop_size[0]) / 2))

        return new_x, new_y

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)
    
class AttentionAttributeDataset(Dataset):
    def __init__(self, scenes_json_file, image_path, max_number_samples) -> None:
        super().__init__()

        preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.samples = []

        with open(scenes_json_file, 'r', encoding='utf-8') as f:
            scenes_metadata = json.load(f)
        for index, scene in enumerate(scenes_metadata['scenes']):
            if index == max_number_samples:
                break

            image = Image.open(image_path + scene['image_filename']).convert('RGB')

            target_object = scene['groups']['target'][0]
            target_x, target_y, _ = scene['objects'][target_object]['pixel_coords']

            color_tensor = self._one_hot_encode(Color, scene['objects'][target_object]['color'])
            shape_tensor = self._one_hot_encode(Shape, scene['objects'][target_object]['shape'])
            size_tensor = self._one_hot_encode(Size, scene['objects'][target_object]['size'])

            self.samples.append(((preprocess(image), color_tensor, shape_tensor, size_tensor),
                                 torch.tensor([target_x, target_y])))

    def _one_hot_encode(self, attribute: Enum, value: str):
        tensor = torch.zeros(len(attribute))
        tensor[attribute[value.upper()].value] = 1

        return tensor
    
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)
