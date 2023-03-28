import json
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights


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
            target_index = indices.index(target_object)
            
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
    